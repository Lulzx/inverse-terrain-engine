//! Persistent grid-parallel worker set (spec §7.7). Grid-parallel mode does not fork-join
//! per step: a fixed set of workers lives as long as the workspace, waits for the next
//! step with a spin-then-park barrier, and claims strip tasks from a shared atomic
//! counter (so a slow core or a busy neighbour process only delays its own strips). A
//! step costs a few atomic round trips instead of a work-stealing scope, and allocates
//! nothing.
//!
//! Results never depend on which thread runs a strip: strips write disjoint rows and the reductions
//! are done afterwards in strip order by the caller (D0).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Spin iterations before a waiting worker parks (≈ tens of µs: longer than the serial
/// part of a step, much shorter than the gap between runs).
const SPIN: u32 = 1 << 14;

type Job = *const (dyn Fn(usize) + Sync + 'static);

struct Shared {
    epoch: AtomicUsize,
    done: AtomicUsize,
    next: AtomicUsize,
    sleeping: AtomicUsize,
    shutdown: AtomicBool,
    panicked: AtomicBool,
    job: UnsafeCell<(Job, usize)>,
    lock: Mutex<()>,
    cv: Condvar,
}

// SAFETY: `job` is written only by the caller while every worker is idle (between the
// completion of one epoch and the publication of the next, ordered by `epoch`/`done`),
// and read by workers only after they observe the new epoch.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct GridPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

fn noop(_: usize) {}

impl GridPool {
    /// `threads` participants in total (the calling thread plus `threads - 1` workers).
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let noop: &'static (dyn Fn(usize) + Sync) = &noop;
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            sleeping: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            job: UnsafeCell::new((noop as Job, 0)),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let workers = (1..threads)
            .map(|p| {
                let sh = shared.clone();
                std::thread::Builder::new().name(format!("itr-grid-{p}")).spawn(move || worker(&sh)).expect("spawn grid worker")
            })
            .collect();
        Self { shared, workers }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(0..tasks)` across the worker set and return when every task is done.
    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        let n = self.threads();
        if n == 1 || tasks <= 1 {
            (0..tasks).for_each(f);
            return;
        }
        let sh = &*self.shared;
        // SAFETY: lifetime erasure. Workers dereference the pointer only during this epoch
        // and this function does not return until all of them have reported `done`.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        unsafe { *sh.job.get() = (job, tasks) };
        sh.done.store(0, Relaxed);
        sh.next.store(0, Relaxed);
        sh.epoch.fetch_add(1, SeqCst);
        if sh.sleeping.load(SeqCst) > 0 {
            let _g = sh.lock.lock().unwrap();
            sh.cv.notify_all();
        }
        let caller = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| claim(sh, tasks, f)));
        let mut spins = 0u32;
        while sh.done.load(Acquire) < n - 1 {
            if spins < SPIN {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        if let Err(e) = caller {
            std::panic::resume_unwind(e);
        }
        if sh.panicked.swap(false, Relaxed) {
            panic!("grid worker panicked");
        }
    }
}

fn claim(sh: &Shared, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
    loop {
        let s = sh.next.fetch_add(1, Relaxed);
        if s >= tasks {
            return;
        }
        f(s);
    }
}

fn worker(sh: &Shared) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            let e = sh.epoch.load(Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if spins < SPIN {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let mut g = sh.lock.lock().unwrap();
            sh.sleeping.fetch_add(1, SeqCst);
            while sh.epoch.load(SeqCst) == seen {
                g = sh.cv.wait(g).unwrap();
            }
            sh.sleeping.fetch_sub(1, SeqCst);
        }
        if sh.shutdown.load(Acquire) {
            return;
        }
        // SAFETY: see `Shared` and `GridPool::run`.
        let (job, tasks) = unsafe { *sh.job.get() };
        let f = unsafe { &*job };
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| claim(sh, tasks, f))).is_err() {
            sh.panicked.store(true, Relaxed);
        }
        sh.done.fetch_add(1, Release);
    }
}

impl Drop for GridPool {
    fn drop(&mut self) {
        let sh = &*self.shared;
        sh.shutdown.store(true, Release);
        sh.epoch.fetch_add(1, SeqCst);
        {
            let _g = sh.lock.lock().unwrap();
            sh.cv.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Disjoint mutable access to one buffer from several strip tasks.
pub struct Shards<'a, T> {
    ptr: *mut T,
    len: usize,
    _m: std::marker::PhantomData<&'a mut [T]>,
}

unsafe impl<T: Send> Sync for Shards<'_, T> {}
unsafe impl<T: Send> Send for Shards<'_, T> {}

impl<'a, T> Shards<'a, T> {
    pub fn new(s: &'a mut [T]) -> Self {
        Self { ptr: s.as_mut_ptr(), len: s.len(), _m: std::marker::PhantomData }
    }
    /// Chunk `i` of size `chunk` (the last may be shorter).
    ///
    /// # Safety
    /// Each chunk index must be taken by at most one task at a time.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn chunk(&self, i: usize, chunk: usize) -> &mut [T] {
        let a = (i * chunk).min(self.len);
        let b = (a + chunk).min(self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(a), b - a) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_task_once_and_survives_parking() {
        let pool = GridPool::new(4);
        for round in 0..200 {
            let hits: Vec<AtomicUsize> = (0..13).map(|_| AtomicUsize::new(0)).collect();
            pool.run(13, &|s| {
                hits[s].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
            if round % 50 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(5)); // let workers park
            }
        }
    }

    #[test]
    fn shards_are_disjoint() {
        let pool = GridPool::new(3);
        let mut v = vec![0usize; 100];
        let sh = Shards::new(&mut v);
        pool.run(10, &|s| unsafe { sh.chunk(s, 10) }.iter_mut().for_each(|x| *x = s));
        assert!(v.iter().enumerate().all(|(i, &x)| x == i / 10));
    }
}
