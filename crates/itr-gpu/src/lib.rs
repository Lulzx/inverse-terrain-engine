//! Optional GPU backend (spec 7.8): wgpu/WGSL, f32, baseline two-dispatch variant
//! (update + 1-workgroup finalize per step), candidate batching along dispatch z, many
//! steps per submit, device-side dt, double-single time, device-side exact rain integral.
//!
//! Supported: `FaceMode::Signed` (level 1) terrains, Wall and Transmissive boundaries,
//! constant infiltration capacity, piecewise-linear rain, monitored running maxima.
//! Unsupported (returns `Err`): Crest mode (coarse levels), Stage and Inflow boundaries,
//! baseline checkpoints/warm start, ledger rows, time-of-peak, `skip_dry`, and the
//! SolverParams extensions for locked dt / hard CFL / inflow scaling (ignored).

use itr_core::model::MonitorSet;
use itr_hydro::terrain::{FaceMode, Ghost};
use itr_hydro::{PreparedTerrain, SolverParams};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shader.wgsl");
const WG: u32 = 16;
const CTL_WORDS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    nx: u32,
    ny: u32,
    ncand: u32,
    n_mon: u32,
    g: f32,
    h_dry: f32,
    heps2: f32,
    q_min: f32,
    inv_dx: f32,
    inv_dy: f32,
    dx: f32,
    dy: f32,
    cfl: f32,
    dt_max: f32,
    duration: f32,
    sync: f32,
    sync_tol: f32,
    infil_cap: f32,
    n_sync: u32,
    n_rain: u32,
    n_cells: u32,
    has_mon: u32,
    pad: [u32; 2],
}

/// Result of one candidate run.
#[derive(Clone, Debug)]
pub struct GpuRun {
    pub monitor_max: Vec<f64>,
    pub steps: u64,
    pub t_end: f64,
    pub volume_m3: f64,
    /// Final interior depth, row-major `nx * ny`.
    pub depth: Vec<f32>,
    /// Non-finite or beyond-roundoff negative depth was detected (CPU: `Unhealthy` abort).
    pub bad: bool,
}

pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    layout: wgpu::BindGroupLayout,
    p_update: wgpu::ComputePipeline,
    p_final: wgpu::ComputePipeline,
    /// Steps encoded per command buffer (host polls status only between submits).
    pub steps_per_submit: u32,
    /// Total GPU cell-updates executed by the last `run_batch` (active candidates only).
    pub last_cell_updates: u64,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl GpuSolver {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("itr-gpu"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("request_device: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("itr-swe"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        // 1 state(rw) 2 dzx 3 dzy 4 cellp 5 bc | 6 ctl(rw) 7 smaxbad(rw) 8 mon(rw) | 9 mon_map 10 active 11 rain
        for (b, ro) in [(1, false), (2, true), (3, true), (4, true), (5, true), (6, false), (7, false), (8, false), (9, true), (10, true), (11, true)] {
            entries.push(storage_entry(b, ro));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (p_update, p_final) = (mk("update"), mk("finalize"));
        Ok(Self { device, queue, info, layout, p_update, p_final, steps_per_submit: 64, last_cell_updates: 0 })
    }

    pub fn adapter_info(&self) -> String {
        format!("{} ({:?}, {:?}, driver '{}' {})", self.info.name, self.info.backend, self.info.device_type, self.info.driver, self.info.driver_info)
    }

    fn buf_init<T: bytemuck::Pod>(&self, label: &str, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let bytes: &[u8] = if data.is_empty() { &[0u8; 16] } else { bytemuck::cast_slice(data) };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
    }

    fn map_read(&self, staging: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>, String> {
        let slice = staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("poll: {e}"))?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| format!("map: {e}"))?;
        let v = slice.get_mapped_range().map_err(|e| format!("range: {e}"))?.to_vec();
        staging.unmap();
        Ok(v)
    }

    /// Run a batch of candidates sharing one `TerrainBase<f32>` (same `Arc`).
    pub fn run_batch(&mut self, p: &SolverParams, terrains: &[&PreparedTerrain<f32>], monitors: &MonitorSet) -> Result<Vec<GpuRun>, String> {
        let nc = terrains.len();
        if nc == 0 {
            return Ok(vec![]);
        }
        let base = terrains[0].base.clone();
        for t in terrains {
            if !Arc::ptr_eq(&t.base, &base) {
                return Err("all candidates must share the same TerrainBase".into());
            }
        }
        if base.mode != FaceMode::Signed {
            return Err("GPU backend supports FaceMode::Signed (level 1) only; Crest mode is unsupported".into());
        }
        let (nx, ny) = (base.nx(), base.ny());
        let n = nx * ny;
        // Boundary codes: north | south | west | east.
        let code = |g: &Ghost| match g {
            Ghost::Wall => Ok(0u32),
            Ghost::Transmissive => Ok(1u32),
            Ghost::Stage(_) => Err("Stage boundaries are not supported on the GPU backend".to_string()),
            Ghost::Inflow(..) => Err("Inflow boundaries are not supported on the GPU backend".to_string()),
        };
        let b = &base.boundaries;
        let mut bc: Vec<u32> = Vec::with_capacity(2 * nx + 2 * ny);
        for g in b.north.iter().chain(&b.south).chain(&b.west).chain(&b.east) {
            bc.push(code(g)?);
        }
        // Face jumps per candidate.
        let (sx, sy) = (ny * (nx + 1), (ny + 1) * nx);
        let mut dzx = vec![0f32; nc * sx];
        let mut dzy = vec![0f32; nc * sy];
        for (c, t) in terrains.iter().enumerate() {
            for r in 0..ny {
                let row = t.fx[0].row(r + 1);
                for f in 1..nx {
                    dzx[c * sx + r * (nx + 1) + f] = row[f];
                }
            }
            for f in 1..ny {
                let row = t.fy[0].row(f);
                for col in 0..nx {
                    dzy[c * sy + f * nx + col] = row[col + 1];
                }
            }
        }
        let mut cellp = vec![[0f32; 2]; n];
        for r in 0..ny {
            let (w, g) = (base.wet.row(r + 1), base.gn2.row(r + 1));
            for col in 0..nx {
                cellp[r * nx + col] = [w[col + 1], g[col + 1]];
            }
        }
        // Monitors.
        let n_mon = monitors.cells.len();
        let mut mon_map = vec![0u32; n];
        for (k, &c) in monitors.cells.iter().enumerate() {
            if (c as usize) >= n {
                return Err("monitor cell out of range".into());
            }
            mon_map[c as usize] = k as u32 + 1;
        }
        // Initial state (same rule as the CPU init_state) and initial dt.
        let mut state = vec![0f32; 2 * 3 * nc * n];
        let mut ctl = vec![0u32; nc * CTL_WORDS];
        for (c, t) in terrains.iter().enumerate() {
            let z = t.z_level();
            let mut s0 = 0f64;
            for k in 0..n {
                if base.wall.data[k] {
                    continue;
                }
                let mut d = p.initial_depth;
                if let Some(s) = p.initial_stage {
                    d = d.max(s - base.fine.z_ref - z[k]);
                }
                let h = (d.max(0.0)) as f32;
                state[c * n + k] = h; // half 0, q 0
                if (h as f64) > p.h_dry {
                    let cc = (base.g * h as f64).sqrt();
                    s0 = s0.max(cc / base.dx + cc / base.dy);
                }
            }
            let dt0 = if s0 > 0.0 { (p.cfl / s0).min(p.dt_max_s) } else { p.dt_max_s };
            ctl[c * CTL_WORDS + 2] = (dt0 as f32).to_bits();
        }
        let n_sync = (p.duration_s / p.sync_interval_s).ceil() as u32;
        let rain: Vec<[f32; 2]> = p.rain.pts.iter().map(|&(t, r)| [t as f32, r as f32]).collect();
        let gl = Globals {
            nx: nx as u32,
            ny: ny as u32,
            ncand: nc as u32,
            n_mon: n_mon as u32,
            g: base.g as f32,
            h_dry: p.h_dry as f32,
            heps2: (p.h_eps * p.h_eps) as f32,
            q_min: 1e-30,
            inv_dx: (1.0 / base.dx) as f32,
            inv_dy: (1.0 / base.dy) as f32,
            dx: base.dx as f32,
            dy: base.dy as f32,
            cfl: p.cfl as f32,
            dt_max: p.dt_max_s as f32,
            duration: p.duration_s as f32,
            sync: p.sync_interval_s as f32,
            sync_tol: (1e-6 * p.sync_interval_s) as f32,
            infil_cap: p.infil_capacity as f32,
            n_sync,
            n_rain: rain.len() as u32,
            n_cells: n as u32,
            has_mon: (n_mon > 0) as u32,
            pad: [0; 2],
        };
        use wgpu::BufferUsages as U;
        let st = U::STORAGE;
        let b_g = self.buf_init("globals", &[gl], U::UNIFORM);
        let b_state = self.buf_init("state", &state, st | U::COPY_SRC);
        let b_dzx = self.buf_init("dzx", &dzx, st);
        let b_dzy = self.buf_init("dzy", &dzy, st);
        let b_cell = self.buf_init("cellp", &cellp, st);
        let b_bc = self.buf_init("bc", &bc, st);
        let b_ctl = self.buf_init("ctl", &ctl, st | U::COPY_SRC);
        let b_sb = self.buf_init("smaxbad", &vec![0u32; nc * 2], st);
        let b_mon = self.buf_init("mon", &vec![0u32; (nc * n_mon).max(1)], st | U::COPY_SRC);
        let b_mm = self.buf_init("mon_map", &mon_map, st);
        let b_act = self.buf_init("active", &(0..nc as u32).collect::<Vec<_>>(), st | U::COPY_DST);
        let b_rain = self.buf_init("rain", &rain, st);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_g.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_dzx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_dzy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_cell.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_bc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b_ctl.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b_sb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: b_mon.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: b_mm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: b_act.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: b_rain.as_entire_binding() },
            ],
        });
        let ctl_bytes = (nc * CTL_WORDS * 4) as u64;
        let stage_ctl = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: ctl_bytes, usage: U::MAP_READ | U::COPY_DST, mapped_at_creation: false });
        let (gx, gy) = ((nx as u32).div_ceil(WG), (ny as u32).div_ceil(WG));
        let mut active: Vec<u32> = (0..nc as u32).collect();
        let mut cell_updates = 0u64;
        let mut last_active_len = nc;
        let k = self.steps_per_submit.max(1);
        let mut guard = 0u64;
        while !active.is_empty() {
            if active.len() != last_active_len || guard == 0 {
                self.queue.write_buffer(&b_act, 0, bytemuck::cast_slice(&active));
                last_active_len = active.len();
            }
            let na = active.len() as u32;
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &bg, &[]);
                for _ in 0..k {
                    pass.set_pipeline(&self.p_final);
                    pass.dispatch_workgroups(na, 1, 1);
                    pass.set_pipeline(&self.p_update);
                    pass.dispatch_workgroups(gx, gy, na);
                }
            }
            enc.copy_buffer_to_buffer(&b_ctl, 0, &stage_ctl, 0, ctl_bytes);
            self.queue.submit([enc.finish()]);
            let raw = self.map_read(&stage_ctl, ctl_bytes)?;
            let words: &[u32] = bytemuck::cast_slice(&raw);
            cell_updates += k as u64 * na as u64 * n as u64;
            active.retain(|&c| words[c as usize * CTL_WORDS + 14] & 1 == 0);
            guard += 1;
            if guard > 10_000_000 {
                return Err("run did not terminate".into());
            }
        }
        self.last_cell_updates = cell_updates;
        // Final read-back.
        let raw_ctl = {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&b_ctl, 0, &stage_ctl, 0, ctl_bytes);
            self.queue.submit([enc.finish()]);
            self.map_read(&stage_ctl, ctl_bytes)?
        };
        let ctlw: &[u32] = bytemuck::cast_slice(&raw_ctl);
        let half_bytes = (nc * n * 4) as u64;
        let stage_h = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 2 * half_bytes, usage: U::MAP_READ | U::COPY_DST, mapped_at_creation: false });
        let half_off = (3 * nc * n * 4) as u64; // start of half 1
        let stage_m = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: ((nc * n_mon).max(1) * 4) as u64, usage: U::MAP_READ | U::COPY_DST, mapped_at_creation: false });
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&b_state, 0, &stage_h, 0, half_bytes);
            enc.copy_buffer_to_buffer(&b_state, half_off, &stage_h, half_bytes, half_bytes);
            enc.copy_buffer_to_buffer(&b_mon, 0, &stage_m, 0, ((nc * n_mon).max(1) * 4) as u64);
            self.queue.submit([enc.finish()]);
        }
        let hraw = self.map_read(&stage_h, 2 * half_bytes)?;
        let hs: &[f32] = bytemuck::cast_slice(&hraw);
        let mraw = self.map_read(&stage_m, ((nc * n_mon).max(1) * 4) as u64)?;
        let ms: &[u32] = bytemuck::cast_slice(&mraw);
        let area = base.cell_area();
        let mut out = Vec::with_capacity(nc);
        for c in 0..nc {
            let w = &ctlw[c * CTL_WORDS..(c + 1) * CTL_WORDS];
            let parity = w[12] as usize;
            let depth: Vec<f32> = hs[parity * nc * n + c * n..parity * nc * n + (c + 1) * n].to_vec();
            let vol: f64 = depth.iter().map(|&v| v as f64).sum::<f64>() * area;
            let bad = (w[14] & 2) != 0 || depth.iter().any(|v| !v.is_finite());
            out.push(GpuRun {
                monitor_max: (0..n_mon).map(|m| f32::from_bits(ms[c * n_mon + m]) as f64).collect(),
                steps: w[11] as u64,
                t_end: f32::from_bits(w[0]) as f64 + f32::from_bits(w[1]) as f64,
                volume_m3: vol,
                depth,
                bad,
            });
        }
        Ok(out)
    }
}
