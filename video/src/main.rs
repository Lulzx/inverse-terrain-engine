use fframes::cli::clap; // the derive below expands to `clap::...`
use fframes::{EncoderOptions, RenderOptions, StaticMediaProvider, cli};
use fframes_skia_renderer::{SkiaFFramesRenderer, SkiaPipelineConcurrencyPolicy, SkiaPipelineConfig, metal::SkiaMetalCtx};
use std::path::PathBuf;
use std::process::ExitCode;
use video::data::RunData;
use video::{Flood2d, Flood3d, HEIGHT, VideoMedia, WIDTH};

/// Flags of this video next to the standard ones of `fframes::cli` (render, frame, strip,
/// inspect, ...). Run `cargo run --release -- --help`.
#[derive(Debug, clap::Args)]
struct VideoArgs {
    /// Directory written by `scripts/video_data.py`.
    #[arg(long, default_value = "../runs/video/rolling-hills", global = true)]
    data: PathBuf,
    /// Vertical exaggeration of terrain and water (1 = true scale).
    #[arg(long, default_value_t = 1.0, global = true)]
    vert_exag: f32,
    /// `3d`: perspective views of both terrains; `2d`: maps and charts.
    #[arg(long, default_value = "3d", global = true)]
    layout: String,
    /// Shader debug view: 0 off, 1 pixel coordinates, 2 elevation texture, 3 water depth.
    #[arg(long, default_value_t = 0.0, global = true)]
    debug: f32,
}

fn main() -> ExitCode {
    let args = cli::parse::<VideoArgs>();
    let media = VideoMedia::prepare().expect("media");
    let data = match RunData::load(&args.app.data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let gpu = SkiaMetalCtx::new(WIDTH, HEIGHT).expect("GPU context");
    let renderer = || {
        SkiaFFramesRenderer::new_metal(
            &gpu,
            SkiaPipelineConfig { concurrency_policy: SkiaPipelineConcurrencyPolicy::MaxPerformance, ..Default::default() },
        )
        .expect("skia renderer")
    };
    let options = RenderOptions {
        media: Some(&media),
        // libx264: the VideoToolbox encoder stalled the Skia pipeline here (fframes 1.2.1-rc.14).
        video_encoder_options: EncoderOptions {
            preferred_encoder: Some("libx264"),
            codec_params: Some(&[("crf", "18"), ("preset", "veryfast")]),
            ..Default::default()
        },
        ..Default::default()
    };
    match args.app.layout.as_str() {
        "2d" => {
            let video = Flood2d::new(&data);
            cli::new(&video, options).args(args).backend(renderer()).preview(fframes_native_player::cli_preview).run()
        }
        _ => {
            let mut video = Flood3d::new(&data, args.app.vert_exag);
            video.debug = args.app.debug;
            cli::new(&video, options).args(args).backend(renderer()).preview(fframes_native_player::cli_preview).run()
        }
    }
}
