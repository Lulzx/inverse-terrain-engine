use itr_hydro::validation::suite;

#[test]
fn validation_suite_f64() {
    let r = suite::<f64>();
    for c in &r {
        println!("{:>24} {:<6} {:>12.3e} <= {:>9.1e} {} {}", c.name, c.passed, c.value, c.threshold, c.precision, c.notes);
    }
    assert!(r.iter().all(|c| c.passed), "failures: {:?}", r.iter().filter(|c| !c.passed).map(|c| &c.name).collect::<Vec<_>>());
}

#[test]
fn validation_suite_f32() {
    let r = suite::<f32>();
    for c in &r {
        println!("{:>24} {:<6} {:>12.3e} <= {:>9.1e} {} {}", c.name, c.passed, c.value, c.threshold, c.precision, c.notes);
    }
    assert!(r.iter().all(|c| c.passed), "failures: {:?}", r.iter().filter(|c| !c.passed).map(|c| &c.name).collect::<Vec<_>>());
}
