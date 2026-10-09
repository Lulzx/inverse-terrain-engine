//! Property tests (spec §9.1.1): lake at rest on random terrains (with wet/dry fronts and
//! walls), and `h ≥ 0` with ledger closure under random rain and random edits, for the
//! fused HLL kernel and the local-inertial screening kernel.

use itr_core::design::TerrainEdit;
use itr_core::model::{MonitorSet, NoopObserver};
use itr_core::scenario::Hyetograph;
use itr_hydro::validation::{params, terrain};
use itr_hydro::{KernelKind, PreparedTerrain, Solver, Workspace};
use proptest::prelude::*;

#[derive(Debug, Clone)]
struct Bed {
    nx: usize,
    ny: usize,
    amp: [f64; 3],
    freq: [f64; 3],
    step: f64,
    wall: Option<(usize, usize)>,
}

fn bed() -> impl Strategy<Value = Bed> {
    (12usize..40, 10usize..32, prop::array::uniform3(0.0..0.8f64), prop::array::uniform3(2.0..15.0f64), -0.5..0.5f64, prop::option::of((2usize..8, 2usize..8)))
        .prop_map(|(nx, ny, amp, freq, step, wall)| Bed { nx, ny, amp, freq, step, wall })
}

fn z(b: &Bed, x: f64, y: f64) -> f64 {
    10.0 + b.amp[0] * (x / b.freq[0]).sin() + b.amp[1] * (y / b.freq[1]).cos() + b.amp[2] * ((x + y) / b.freq[2]).sin() + if x > b.nx as f64 { b.step } else { 0.0 }
}

fn kernel(li: bool) -> KernelKind {
    if li { KernelKind::LocalInertial } else { KernelKind::Fused }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    #[test]
    fn lake_at_rest_random_terrain(b in bed(), stage in 9.0..11.5f64, li in any::<bool>()) {
        let w = b.wall;
        let base = terrain::<f64>(b.nx, b.ny, 2.0, |x, y| z(&b, x, y), |c, r| w.is_some_and(|(wc, wr)| c == wc && r >= wr), 0.03, &[]);
        let mut p = params(60.0, 30.0);
        p.kernel = kernel(li);
        p.initial_stage = Some(stage);
        let terr = PreparedTerrain::new(base.clone());
        let mut ws = Workspace::<f64>::new(base.layout, &MonitorSet::default(), 1, 32);
        let s = Solver::<f64>::new(p).run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
        prop_assert!(s.aborted.is_none());
        let l = ws.layout;
        let mut q: f64 = 0.0;
        for j in 1..=l.ny {
            for i in 1..=l.nx {
                q = q.max(ws.qx[l.at(i, j)].abs()).max(ws.qy[l.at(i, j)].abs());
            }
        }
        prop_assert!(q < 1e-10, "lake at rest discharge {q}");
    }

    #[test]
    fn positivity_and_ledger_under_random_rain_and_edits(
        b in bed(),
        rain in prop::collection::vec(0.0..150.0f64, 1..5),
        dz in prop::collection::vec(-0.6..0.6f64, 16),
        at in (0usize..8, 0usize..6),
        li in any::<bool>(),
        threads in 1usize..4,
    ) {
        let w = b.wall;
        let base = terrain::<f64>(b.nx, b.ny, 2.0, |x, y| z(&b, x, y), |c, r| w.is_some_and(|(wc, wr)| c == wc && r >= wr), 0.03, &[]);
        let mut p = params(400.0, 50.0);
        p.kernel = kernel(li);
        let hy: Vec<[f64; 2]> = rain.iter().enumerate().map(|(k, r)| [k as f64 * 80.0, *r]).collect();
        p.rain = Hyetograph::from_mm_h(&hy);
        let mut terr = PreparedTerrain::new(base.clone());
        terr.apply_edit(&TerrainEdit { c0: at.0, r0: at.1, w: 4, h: 4, dz: dz.clone(), ..Default::default() });
        let mut ws = Workspace::<f64>::new(base.layout, &MonitorSet::default(), threads, 5);
        let s = Solver::<f64>::new(p).run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
        prop_assert!(s.aborted.is_none());
        let l = ws.layout;
        let hmin = (1..=l.ny).flat_map(|j| (1..=l.nx).map(move |i| (i, j))).map(|(i, j)| ws.h[l.at(i, j)]).fold(f64::MAX, f64::min);
        prop_assert!(hmin >= -1e-10, "negative depth {hmin}");
        prop_assert!(s.ledger.rel_residual < 1e-10, "mass residual {}", s.ledger.rel_residual);
    }
}
