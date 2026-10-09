//! Civil-CAD export (spec §19.4): LandXML 1.2 TIN surfaces and DXF R12 grading sheets.

use itr_core::raster::Raster;
use serde_json::Value;
use std::fmt::Write;

/// Outer ring of a Polygon feature as (x, y) pairs, without the closing duplicate.
fn ring(feature: &Value) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = feature["geometry"]["coordinates"][0]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?)))
                .collect()
        })
        .unwrap_or_default();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    pts
}

fn features(earthworks: &Value) -> &[Value] {
    earthworks["features"].as_array().map(|v| v.as_slice()).unwrap_or(&[])
}

/// Window (c0, r0, c1, r1) inclusive of cells where |delta| > 0, grown by `margin`.
fn window(delta: &Raster, margin: usize) -> (usize, usize, usize, usize) {
    let (mut c0, mut r0, mut c1, mut r1) = (usize::MAX, usize::MAX, 0, 0);
    let mut any = false;
    for r in 0..delta.ny {
        for c in 0..delta.nx {
            let d = delta.get(c, r);
            if d.is_finite() && d.abs() > 0.0 {
                any = true;
                c0 = c0.min(c);
                r0 = r0.min(r);
                c1 = c1.max(c);
                r1 = r1.max(r);
            }
        }
    }
    if !any {
        return (0, 0, delta.nx.saturating_sub(1), delta.ny.saturating_sub(1));
    }
    (
        c0.saturating_sub(margin),
        r0.saturating_sub(margin),
        (c1 + margin).min(delta.nx - 1),
        (r1 + margin).min(delta.ny - 1),
    )
}

fn surface(out: &mut String, name: &str, rs: &Raster, win: (usize, usize, usize, usize)) {
    let (c0, r0, c1, r1) = win;
    let w = c1 - c0 + 1;
    let h = r1 - r0 + 1;
    let mut ids = vec![0usize; w * h];
    let mut n = 0;
    let _ = writeln!(out, "      <Surface name=\"{name}\">\n        <Definition surfType=\"TIN\">\n          <Pnts>");
    for r in r0..=r1 {
        for c in c0..=c1 {
            let z = rs.get(c, r);
            if !z.is_finite() {
                continue;
            }
            n += 1;
            ids[(r - r0) * w + (c - c0)] = n;
            let (x, y) = rs.geo.cell_center(c, r);
            // LandXML order: northing easting elevation
            let _ = writeln!(out, "            <P id=\"{n}\">{y:.4} {x:.4} {z:.4}</P>");
        }
    }
    out.push_str("          </Pnts>\n          <Faces>\n");
    for r in 0..h.saturating_sub(1) {
        for c in 0..w.saturating_sub(1) {
            let tl = ids[r * w + c];
            let tr = ids[r * w + c + 1];
            let bl = ids[(r + 1) * w + c];
            let br = ids[(r + 1) * w + c + 1];
            if tl == 0 || tr == 0 || bl == 0 || br == 0 {
                continue;
            }
            // counter-clockwise (north-up): NW, SW, SE and NW, SE, NE
            let _ = writeln!(out, "            <F>{tl} {bl} {br}</F>");
            let _ = writeln!(out, "            <F>{tl} {br} {tr}</F>");
        }
    }
    out.push_str("          </Faces>\n        </Definition>\n      </Surface>\n");
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// LandXML 1.2 document with design and existing TIN surfaces plus earthwork outlines.
pub fn landxml(
    before: &Raster,
    after: &Raster,
    delta: &Raster,
    earthworks: &Value,
    epsg: Option<u32>,
    margin: usize,
) -> String {
    let win = window(delta, margin);
    let mut o = String::new();
    o.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    o.push_str("<LandXML xmlns=\"http://www.landxml.org/schema/LandXML-1.2\" version=\"1.2\" language=\"English\">\n");
    o.push_str("  <Units>\n    <Metric linearUnit=\"meter\" areaUnit=\"squareMeter\" volumeUnit=\"cubicMeter\" temperatureUnit=\"celsius\" pressureUnit=\"HPA\"/>\n  </Units>\n");
    if let Some(code) = epsg {
        let _ = writeln!(o, "  <CoordinateSystem epsgCode=\"{code}\"/>");
    }
    o.push_str("  <Surfaces>\n");
    surface(&mut o, "design", after, win);
    surface(&mut o, "existing", before, win);
    o.push_str("  </Surfaces>\n  <PlanFeatures>\n");
    for (i, f) in features(earthworks).iter().enumerate() {
        let pts = ring(f);
        if pts.len() < 2 {
            continue;
        }
        let kind = f["properties"]["kind"].as_str().unwrap_or("earthwork");
        let idx = f["properties"]["index"].as_i64().unwrap_or(i as i64);
        let _ = writeln!(o, "    <PlanFeature name=\"{}_{idx}\">\n      <CoordGeom>", esc(kind));
        for k in 0..pts.len() {
            let (x0, y0) = pts[k];
            let (x1, y1) = pts[(k + 1) % pts.len()];
            let _ = writeln!(
                o,
                "        <Line><Start>{y0:.4} {x0:.4}</Start><End>{y1:.4} {x1:.4}</End></Line>"
            );
        }
        o.push_str("      </CoordGeom>\n    </PlanFeature>\n");
    }
    o.push_str("  </PlanFeatures>\n</LandXML>\n");
    o
}

fn pair(o: &mut String, code: i32, v: impl std::fmt::Display) {
    let _ = writeln!(o, "{code}\n{v}");
}

fn text(o: &mut String, layer: &str, x: f64, y: f64, h: f64, s: &str) {
    pair(o, 0, "TEXT");
    pair(o, 8, layer);
    pair(o, 10, format!("{x:.4}"));
    pair(o, 20, format!("{y:.4}"));
    pair(o, 30, "0.0000");
    pair(o, 40, format!("{h:.4}"));
    pair(o, 1, s);
}

/// DXF R12 ASCII with earthwork outlines, edited-cell points and a cut/fill summary.
pub fn dxf(after: &Raster, delta: &Raster, earthworks: &Value) -> String {
    let mut o = String::new();
    let layers = [("ITR_EARTHWORKS", 3), ("ITR_CUT", 1), ("ITR_FILL", 5), ("ITR_SPOT", 7)];
    pair(&mut o, 0, "SECTION");
    pair(&mut o, 2, "HEADER");
    pair(&mut o, 9, "$ACADVER");
    pair(&mut o, 1, "AC1009");
    pair(&mut o, 9, "$INSUNITS");
    pair(&mut o, 70, 6);
    pair(&mut o, 0, "ENDSEC");
    pair(&mut o, 0, "SECTION");
    pair(&mut o, 2, "TABLES");
    pair(&mut o, 0, "TABLE");
    pair(&mut o, 2, "LAYER");
    pair(&mut o, 70, layers.len());
    for (name, color) in layers {
        pair(&mut o, 0, "LAYER");
        pair(&mut o, 2, name);
        pair(&mut o, 70, 0);
        pair(&mut o, 62, color);
        pair(&mut o, 6, "CONTINUOUS");
    }
    pair(&mut o, 0, "ENDTAB");
    pair(&mut o, 0, "ENDSEC");
    pair(&mut o, 0, "SECTION");
    pair(&mut o, 2, "ENTITIES");

    let th = after.geo.dx.abs().max(after.geo.dy.abs());
    for (i, f) in features(earthworks).iter().enumerate() {
        let pts = ring(f);
        if pts.len() < 2 {
            continue;
        }
        pair(&mut o, 0, "POLYLINE");
        pair(&mut o, 8, "ITR_EARTHWORKS");
        pair(&mut o, 66, 1);
        pair(&mut o, 10, "0.0000");
        pair(&mut o, 20, "0.0000");
        pair(&mut o, 30, "0.0000");
        pair(&mut o, 70, 1);
        for &(x, y) in &pts {
            pair(&mut o, 0, "VERTEX");
            pair(&mut o, 8, "ITR_EARTHWORKS");
            pair(&mut o, 10, format!("{x:.4}"));
            pair(&mut o, 20, format!("{y:.4}"));
            pair(&mut o, 30, "0.0000");
        }
        pair(&mut o, 0, "SEQEND");
        pair(&mut o, 8, "ITR_EARTHWORKS");
        let p = &f["properties"];
        let n = pts.len() as f64;
        let cx = p["cx"].as_f64().unwrap_or(pts.iter().map(|q| q.0).sum::<f64>() / n);
        let cy = p["cy"].as_f64().unwrap_or(pts.iter().map(|q| q.1).sum::<f64>() / n);
        let kind = p["kind"].as_str().unwrap_or("earthwork");
        let idx = p["index"].as_i64().unwrap_or(i as i64);
        let hm = p["height_m"].as_f64().unwrap_or(0.0);
        text(&mut o, "ITR_EARTHWORKS", cx, cy, th, &format!("{kind} #{idx} h={hm:.2} m"));
    }

    let (mut cut, mut fill) = (0.0f64, 0.0f64);
    let area = after.geo.cell_area().abs();
    for r in 0..delta.ny {
        for c in 0..delta.nx {
            let d = delta.get(c, r);
            if !d.is_finite() {
                continue;
            }
            cut += (-d).max(0.0) * area;
            fill += d.max(0.0) * area;
            let z = after.get(c, r);
            if d.abs() > 0.005 && z.is_finite() {
                let (x, y) = after.geo.cell_center(c, r);
                pair(&mut o, 0, "POINT");
                pair(&mut o, 8, if d < 0.0 { "ITR_CUT" } else { "ITR_FILL" });
                pair(&mut o, 10, format!("{x:.4}"));
                pair(&mut o, 20, format!("{y:.4}"));
                pair(&mut o, 30, format!("{z:.4}"));
            }
        }
    }
    // summary block below the grid's south-west corner
    let x = after.geo.origin_x;
    let y = after.geo.origin_y - after.ny as f64 * after.geo.dy - 2.0 * th;
    text(&mut o, "ITR_SPOT", x, y, th, &format!("Total cut: {cut:.2} m3"));
    text(&mut o, "ITR_SPOT", x, y - 1.5 * th, th, &format!("Total fill: {fill:.2} m3"));
    pair(&mut o, 0, "ENDSEC");
    pair(&mut o, 0, "EOF");
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use itr_core::raster::GeoTransform;
    use serde_json::json;

    fn fixture() -> (Raster, Raster, Raster, Value) {
        let geo = GeoTransform { origin_x: 100.0, origin_y: 205.0, dx: 2.0, dy: 2.0 };
        let mut before = Raster::new(6, 5, geo.clone(), 10.0);
        for r in 0..5 {
            for c in 0..6 {
                before.set(c, r, 10.0 + c as f64 * 0.1);
            }
        }
        let mut delta = Raster::new(6, 5, geo.clone(), 0.0);
        for r in 1..3 {
            for c in 2..4 {
                delta.set(c, r, 0.5);
            }
        }
        let mut after = before.clone();
        for i in 0..after.data.len() {
            after.data[i] += delta.data[i];
        }
        let fc = json!({"type":"FeatureCollection","features":[{
            "type":"Feature",
            "geometry":{"type":"Polygon","coordinates":[[[104.0,199.0],[108.0,199.0],[108.0,203.0],[104.0,203.0],[104.0,199.0]]]},
            "properties":{"index":3,"kind":"berm","cx":106.0,"cy":201.0,"length_m":4.0,"width_m":4.0,"angle_deg":0.0,"height_m":0.5}
        }]});
        (before, after, delta, fc)
    }

    fn count(s: &str, pat: &str) -> usize {
        s.matches(pat).count()
    }

    #[test]
    fn landxml_structure_and_ordering() {
        let (b, a, d, fc) = fixture();
        // window: cols 2..3, rows 1..2, margin 1 -> cols 1..4, rows 0..3 = 4x4
        let x = landxml(&b, &a, &d, &fc, Some(32643), 1);
        for t in ["Surface name", "<Pnts>", "<Faces>", "<PlanFeature "] {
            assert_eq!(count(&x, t), if t == "Surface name" { 2 } else { if t == "<PlanFeature " { 1 } else { 2 } }, "{t}");
        }
        assert_eq!(count(&x, "</Surface>"), 2);
        assert_eq!(count(&x, "</Pnts>"), 2);
        assert_eq!(count(&x, "</Faces>"), 2);
        assert_eq!(count(&x, "</PlanFeature>"), 1);
        assert_eq!(count(&x, "<P id="), 32);
        assert_eq!(count(&x, "<F>"), 2 * 2 * 3 * 3);
        assert_eq!(count(&x, "<Line>"), 4);
        assert!(x.contains("epsgCode=\"32643\""));
        assert!(x.contains("volumeUnit=\"cubicMeter\""));
        // first design point is cell (1,0): E=105, N=204, Z=10.1
        assert!(x.contains("<P id=\"1\">204.0000 103.0000 10.1000</P>"), "{x}");
        // cell (2,1) in design: E=105, N=202, Z=10.2+0.5
        assert!(x.contains("202.0000 105.0000 10.7000"));
        assert!(x.contains("202.0000 105.0000 10.2000")); // existing
        // no-margin, no-epsg
        let y = landxml(&b, &a, &d, &fc, None, 0);
        assert!(!y.contains("CoordinateSystem"));
        assert_eq!(count(&y, "<F>"), 2 * 2);
    }

    #[test]
    fn landxml_skips_nan_and_whole_grid_when_no_delta() {
        let (b, mut a, d, fc) = fixture();
        a.set(1, 1, f64::NAN);
        let zero = Raster::new(6, 5, d.geo.clone(), 0.0);
        let x = landxml(&b, &a, &zero, &fc, None, 0);
        // design: 29 pts; 20 cells, 4 touch (1,1) -> 16 cells -> 32 faces; existing: 30 pts, 40 faces
        assert_eq!(count(&x, "<P id="), 29 + 30);
        assert_eq!(count(&x, "<F>"), 32 + 40);
    }

    #[test]
    fn dxf_content() {
        let (_, a, d, fc) = fixture();
        let s = dxf(&a, &d, &fc);
        assert!(s.starts_with("0\nSECTION"));
        assert!(s.ends_with("0\nEOF\n"));
        for l in ["ITR_EARTHWORKS", "ITR_CUT", "ITR_FILL", "ITR_SPOT"] {
            assert!(s.contains(&format!("2\n{l}\n")), "{l}");
        }
        assert!(s.contains("9\n$INSUNITS\n70\n6\n"));
        assert_eq!(count(&s, "0\nPOINT\n"), 4);
        assert_eq!(count(&s, "0\nPOLYLINE\n"), 1);
        assert_eq!(count(&s, "0\nVERTEX\n"), 4);
        assert_eq!(count(&s, "0\nSEQEND\n"), 1);
        assert!(s.contains("70\n1\n0\nVERTEX"));
        assert!(s.contains("1\nberm #3 h=0.50 m\n"));
        // 4 cells * 0.5 m * 4 m2 = 8 m3 fill
        assert!(s.contains("1\nTotal fill: 8.00 m3\n"));
        assert!(s.contains("1\nTotal cut: 0.00 m3\n"));
        assert_eq!(count(&s, "8\nITR_FILL\n"), 4);
    }
}
