//! Grid-parallel steps allocate nothing on any thread (spec §7.7, §9.1.1): the persistent
//! worker set is reused across steps and runs. Single test in its own binary so the
//! process-wide allocation counter sees no other test.

use itr_core::model::{MonitorSet, NoopObserver};
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_hydro::validation::{params, terrain};
use itr_hydro::{PreparedTerrain, Solver, Workspace};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ON: AtomicBool = AtomicBool::new(false);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GA: Counting = Counting;

#[test]
fn grid_parallel_steps_do_not_allocate() {
    let segs = [BoundarySegment::Inflow { side: Side::West, hydrograph: vec![[0.0, 0.5], [900.0, 1.5]], range_m: Some([20.0, 40.0]) }];
    let b = terrain::<f32>(64, 96, 2.0, |x, y| 0.4 * (x / 9.0).sin() * (y / 7.0).cos() + 0.003 * (128.0 - x), |_, _| false, 0.035, &segs);
    let mut p = params(900.0, 900.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 40.0]]);
    let mon = MonitorSet::new((0..64 * 96).filter(|c| c % 11 == 3).collect());
    let terr = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<f32>::new(b.layout, &mon, 4, 8);
    let s = Solver::<f32>::new(p);
    s.run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap(); // warm-up
    ALLOCS.store(0, Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
    let r = s.run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
    ON.store(false, Ordering::SeqCst);
    let n = ALLOCS.load(Ordering::SeqCst);
    assert!(r.steps > 200);
    assert!(n < 16, "{n} allocations across all threads for {} steps", r.steps);
}
