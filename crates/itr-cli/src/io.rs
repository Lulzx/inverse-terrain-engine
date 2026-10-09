//! File I/O (spec §8.4): single-band projected GeoTIFF read/write with in-house GeoKey
//! handling (no GDAL), GeoJSON polygons via `serde_json`, LZ4 viewer frames.

use itr_core::raster::{GeoTransform, Mask, Polygon, Raster};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::encoder::{colortype, Compression, DeflateLevel, TiffEncoder};
use tiff::tags::Tag;

pub type Res<T> = Result<T, String>;

#[derive(Clone, Debug, Default)]
pub struct GeoInfo {
    /// EPSG code of the projected CRS if declared (ProjectedCSTypeGeoKey 3072).
    pub epsg: Option<u32>,
    /// 1 = projected, 2 = geographic, 3 = geocentric (GTModelTypeGeoKey 1024).
    pub model_type: Option<u16>,
    pub nodata: Option<f64>,
    pub pixel_is_point: bool,
    pub compression: u16,
}

const T_SCALE: u16 = 33550;
const T_TIE: u16 = 33922;
const T_XFORM: u16 = 34264;
const T_GEOKEYS: u16 = 34735;
const T_NODATA: u16 = 42113;

/// Read a single-band GeoTIFF. Nodata (and NaN) cells become NaN in the returned raster.
pub fn read_geotiff(path: &Path) -> Res<(Raster, GeoInfo)> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut d = Decoder::new(BufReader::new(f)).map_err(|e| format!("{}: not a readable TIFF: {e}", path.display()))?;
    d = d.with_limits(Limits::unlimited());
    let (w, h) = d.dimensions().map_err(|e| e.to_string())?;
    let ct = d.colortype().map_err(|e| e.to_string())?;
    if !matches!(ct, tiff::ColorType::Gray(_)) {
        return Err(format!("{}: expected single-band raster, found {ct:?}", path.display()));
    }
    let compression = d.find_tag_unsigned::<u16>(Tag::Compression).ok().flatten().unwrap_or(1);
    let mut info = GeoInfo { compression, ..Default::default() };
    let scale = d.find_tag(Tag::Unknown(T_SCALE)).ok().flatten().map(|v| v.into_f64_vec());
    let tie = d.find_tag(Tag::Unknown(T_TIE)).ok().flatten().map(|v| v.into_f64_vec());
    if d.find_tag(Tag::Unknown(T_XFORM)).ok().flatten().is_some() && scale.is_none() {
        return Err(format!(
            "{}: rotated/sheared ModelTransformation is not supported; resample to north-up: gdalwarp -t_srs <EPSG> in.tif out.tif",
            path.display()
        ));
    }
    let geo = match (scale, tie) {
        (Some(Ok(s)), Some(Ok(t))) if s.len() >= 2 && t.len() >= 6 => {
            GeoTransform { origin_x: t[3] - t[0] * s[0], origin_y: t[4] + t[1] * s[1], dx: s[0], dy: s[1] }
        }
        _ => return Err(format!("{}: missing ModelPixelScale/ModelTiepoint georeferencing", path.display())),
    };
    if let Some(Ok(keys)) = d.find_tag(Tag::Unknown(T_GEOKEYS)).ok().flatten().map(|v| v.into_u16_vec())
        && keys.len() >= 4 {
            let n = keys[3] as usize;
            for k in 0..n {
                let e = &keys[4 + 4 * k..4 + 4 * k + 4];
                if e[1] != 0 {
                    continue; // value stored elsewhere; not needed for the keys we read
                }
                match e[0] {
                    1024 => info.model_type = Some(e[3]),
                    1025 => info.pixel_is_point = e[3] == 2,
                    3072 => info.epsg = Some(e[3] as u32),
                    _ => {}
                }
            }
        }
    if let Some(Ok(s)) = d.find_tag(Tag::Unknown(T_NODATA)).ok().flatten().map(|v| v.into_string()) {
        info.nodata = s.trim().trim_end_matches('\0').parse().ok();
    }
    let img = d.read_image().map_err(|e| {
        format!("{}: cannot decode ({e}). Supported codecs: none/LZW/Deflate. Convert with: gdal_translate -co COMPRESS=DEFLATE in.tif out.tif", path.display())
    })?;
    let mut data: Vec<f64> = match img {
        DecodingResult::F32(v) => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::F64(v) => v,
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::I16(v) => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::U32(v) => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::I32(v) => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::I8(v) => v.into_iter().map(|x| x as f64).collect(),
        _ => return Err(format!("{}: unsupported sample type", path.display())),
    };
    if let Some(nd) = info.nodata {
        for v in data.iter_mut() {
            if *v == nd || (nd.abs() > 1e30 && (*v - nd).abs() <= nd.abs() * 1e-6) {
                *v = f64::NAN;
            }
        }
    }
    let mut geo = geo;
    if info.pixel_is_point {
        geo.origin_x -= 0.5 * geo.dx;
        geo.origin_y += 0.5 * geo.dy;
    }
    Ok((Raster { nx: w as usize, ny: h as usize, geo, data }, info))
}

pub fn epsg_of(crs: &str) -> Option<u32> {
    crs.strip_prefix("EPSG:").and_then(|s| s.parse().ok())
}

/// Write a Float32 Deflate GeoTIFF (striped). NaN is written as nodata −9999.
pub fn write_geotiff(path: &Path, r: &Raster, epsg: Option<u32>) -> Res<()> {
    let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = TiffEncoder::new(BufWriter::new(f))
        .map_err(|e| e.to_string())?
        .with_compression(Compression::Deflate(DeflateLevel::Balanced));
    let mut img = enc.new_image::<colortype::Gray32Float>(r.nx as u32, r.ny as u32).map_err(|e| e.to_string())?;
    let g = &r.geo;
    let e = img.encoder();
    let w = |e: tiff::TiffError| e.to_string();
    e.write_tag(Tag::Unknown(T_SCALE), &[g.dx, g.dy, 0.0][..]).map_err(w)?;
    e.write_tag(Tag::Unknown(T_TIE), &[0.0, 0.0, 0.0, g.origin_x, g.origin_y, 0.0][..]).map_err(w)?;
    let code = epsg.unwrap_or(32767) as u16;
    let keys: [u16; 16] = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, code];
    e.write_tag(Tag::Unknown(T_GEOKEYS), &keys[..]).map_err(w)?;
    e.write_tag(Tag::Unknown(T_NODATA), "-9999").map_err(w)?;
    let data: Vec<f32> = r.data.iter().map(|v| if v.is_finite() { *v as f32 } else { -9999.0 }).collect();
    img.write_data(&data).map_err(|e| e.to_string())
}

/// Polygon features with their properties.
pub struct Feature {
    pub polys: Vec<Polygon>,
    pub props: Value,
}

fn ring(v: &Value) -> Res<Vec<(f64, f64)>> {
    v.as_array()
        .ok_or("bad ring")?
        .iter()
        .map(|p| {
            let a = p.as_array().ok_or("bad position")?;
            Ok((a.first().and_then(Value::as_f64).ok_or("bad x")?, a.get(1).and_then(Value::as_f64).ok_or("bad y")?))
        })
        .collect()
}

fn polygon(v: &Value) -> Res<Polygon> {
    let rings = v.as_array().ok_or("bad polygon")?;
    let exterior = ring(rings.first().ok_or("empty polygon")?)?;
    let holes = rings[1..].iter().map(ring).collect::<Res<_>>()?;
    Ok(Polygon { exterior, holes })
}

fn geometry(g: &Value) -> Res<Vec<Polygon>> {
    match g["type"].as_str() {
        Some("Polygon") => Ok(vec![polygon(&g["coordinates"])?]),
        Some("MultiPolygon") => g["coordinates"].as_array().ok_or("bad multipolygon")?.iter().map(polygon).collect(),
        Some(t) => Err(format!("unsupported geometry type {t} (only Polygon/MultiPolygon)")),
        None => Err("geometry without type".into()),
    }
}

/// Read Polygon/MultiPolygon features (coordinates must be in the DEM's CRS).
pub fn read_geojson(path: &Path) -> Res<Vec<Feature>> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
    let feats: Vec<Value> = match v["type"].as_str() {
        Some("FeatureCollection") => v["features"].as_array().cloned().unwrap_or_default(),
        Some("Feature") => vec![v.clone()],
        _ => vec![json!({"type": "Feature", "geometry": v, "properties": {}})],
    };
    feats
        .iter()
        .map(|f| Ok(Feature { polys: geometry(&f["geometry"]).map_err(|e| format!("{}: {e}", path.display()))?, props: f["properties"].clone() }))
        .collect()
}

pub fn polys_to_geojson(feats: &[(Vec<(f64, f64)>, Value)]) -> Value {
    json!({
        "type": "FeatureCollection",
        "features": feats.iter().map(|(ring, props)| json!({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring.iter().map(|(x, y)| [*x, *y]).collect::<Vec<_>>()]},
            "properties": props,
        })).collect::<Vec<_>>()
    })
}

pub fn write_json(path: &Path, v: &Value) -> Res<()> {
    std::fs::write(path, serde_json::to_string_pretty(v).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn mask_from_geojson(path: &Path, dem: &Raster) -> Res<Mask> {
    let polys: Vec<Polygon> = read_geojson(path)?.into_iter().flat_map(|f| f.polys).collect();
    Ok(itr_core::raster::rasterize(&polys, dem.nx, dem.ny, &dem.geo))
}

/// Viewer frame: depth in u16 millimetres (saturating) + LZ4 (size-prepended block).
pub fn encode_frame(depth: &[f32]) -> (Vec<u8>, bool) {
    let mut sat = false;
    let mut b = Vec::with_capacity(depth.len() * 2);
    for &d in depth {
        let mm = (d as f64 * 1000.0).round();
        if mm > 65535.0 {
            sat = true;
        }
        b.extend_from_slice(&(mm.clamp(0.0, 65535.0) as u16).to_le_bytes());
    }
    (lz4_flex::block::compress_prepend_size(&b), sat)
}

/// Append-only JSON-lines writer.
pub struct Jsonl(BufWriter<File>);
impl Jsonl {
    pub fn create(path: &Path, append: bool) -> Res<Self> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self(BufWriter::new(f)))
    }
    pub fn line<S: serde::Serialize>(&mut self, v: &S) {
        let _ = serde_json::to_writer(&mut self.0, v);
        let _ = self.0.write_all(b"\n");
        let _ = self.0.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geotiff_roundtrip() {
        let geo = GeoTransform { origin_x: 500000.0, origin_y: 4200000.0, dx: 2.0, dy: 2.0 };
        let mut r = Raster::new(7, 5, geo.clone(), 0.0);
        for (i, v) in r.data.iter_mut().enumerate() {
            *v = 100.0 + i as f64 * 0.25;
        }
        r.data[3] = f64::NAN;
        let p = std::env::temp_dir().join(format!("itr_rt_{}.tif", std::process::id()));
        write_geotiff(&p, &r, Some(32643)).unwrap();
        let (q, info) = read_geotiff(&p).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!((q.nx, q.ny), (7, 5));
        assert_eq!(q.geo, geo);
        assert_eq!(info.epsg, Some(32643));
        assert_eq!(info.model_type, Some(1));
        assert!(q.data[3].is_nan());
        assert_eq!(q.data[10], r.data[10]);
    }
}
