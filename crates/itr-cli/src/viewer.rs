//! Static viewer bundle (spec §11): `index.html` + `viewer.js` (embedded at build time)
//! `viewer_data.js` (index, rasters and frames as base64 LZ4). Classic scripts, so the
//! bundle works from `file://`; `--serve` is a tiny std-only HTTP server.

use crate::io::{self, Res};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::Path;

const INDEX_HTML: &str = include_str!("../../../viewer/index.html");
const VIEWER_JS: &str = include_str!("../../../viewer/viewer.js");

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

fn raster_blob(path: &Path) -> Res<Value> {
    let (r, _) = io::read_geotiff(path)?;
    let bytes: Vec<u8> = r.data.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
    Ok(json!(b64(&lz4_flex::block::compress_prepend_size(&bytes))))
}

fn embed_frames(dir: &Path, idx: &mut Value) -> Res<()> {
    let Some(list) = idx["frames"].as_array_mut() else { return Ok(()) };
    for f in list.iter_mut() {
        let name = f["file"].as_str().unwrap_or_default().to_string();
        let b = std::fs::read(dir.join("frames").join(&name)).map_err(|e| format!("{name}: {e}"))?;
        f["data"] = json!(b64(&b));
    }
    if let Some(v) = idx["velocity"]["file"].as_str() {
        let b = std::fs::read(dir.join("frames").join(v)).map_err(|e| format!("{v}: {e}"))?;
        idx["velocity"]["data"] = json!(b64(&b));
    }
    Ok(())
}

pub fn write_bundle(dir: &Path) -> Res<()> {
    let mut idx: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("viewer_index.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut rasters = serde_json::Map::new();
    if let Some(m) = idx["rasters"].as_object() {
        for (k, v) in m {
            let p = dir.join(v.as_str().unwrap_or_default());
            if p.exists() {
                rasters.insert(k.clone(), raster_blob(&p)?);
            }
        }
    }
    for tag in ["baseline", "before", "after"] {
        if !idx[tag].is_null() {
            embed_frames(dir, &mut idx[tag])?;
        }
    }
    if let Ok(m) = std::fs::read_to_string(dir.join("metrics.json")) {
        idx["metrics"] = serde_json::from_str(&m).unwrap_or(Value::Null);
    }
    if let Ok(m) = std::fs::read_to_string(dir.join("earthworks.geojson")) {
        idx["earthworks"] = serde_json::from_str(&m).unwrap_or(Value::Null);
    }
    idx["raster_data"] = Value::Object(rasters);
    let data = format!("window.ITR_DATA = {};\n", serde_json::to_string(&idx).unwrap());
    std::fs::write(dir.join("viewer_data.js"), data).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("viewer.js"), VIEWER_JS).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("index.html"), INDEX_HTML).map_err(|e| e.to_string())
}

/// Minimal static file server (GET only, no directory listing, no path traversal).
pub fn serve(dir: &Path, port: u16) -> Res<()> {
    let l = std::net::TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    eprintln!("serving {} at http://127.0.0.1:{port}/ (Ctrl-C to stop)", dir.display());
    for s in l.incoming() {
        let Ok(mut s) = s else { continue };
        let mut buf = [0u8; 4096];
        let n = s.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..n]);
        let path = req.split_whitespace().nth(1).unwrap_or("/");
        let path = path.split('?').next().unwrap_or("/").trim_start_matches('/');
        let path = if path.is_empty() { "index.html" } else { path };
        let ok = !path.split('/').any(|c| c == ".." || c.is_empty());
        let body = if ok { std::fs::read(dir.join(path)).ok() } else { None };
        let ct = match path.rsplit('.').next() {
            Some("html") => "text/html; charset=utf-8",
            Some("js") => "text/javascript",
            Some("json") => "application/json",
            _ => "application/octet-stream",
        };
        let _ = match body {
            Some(b) => {
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len());
                s.write_all(&b)
            }
            None => s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_known() {
        assert_eq!(super::b64(b"Man"), "TWFu");
        assert_eq!(super::b64(b"Ma"), "TWE=");
        assert_eq!(super::b64(b"M"), "TQ==");
    }
}
