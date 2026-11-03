//! wgpu pipeline + egui paint-callback glue (coordinator-owned).

pub use crate::scene::{CameraMode, CamUniform, Frame3D};
use crate::View3D;
use egui::epaint::PaintCallbackInfo;
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// TEMP diag counter: prints subsystem state every DIAG_EVERY update calls
/// while ALIFE_VIEW3D_DIAG=1; stays silent otherwise.
const DIAG_EVERY: u32 = 120;
static DIAG_N: AtomicU32 = AtomicU32::new(0);

fn diag_tick(view: &View3D) {
    if std::env::var("ALIFE_VIEW3D_DIAG").map(|v| v == "1").unwrap_or(false)
        && DIAG_N.fetch_add(1, Ordering::Relaxed) % DIAG_EVERY == 0
    {
        let frame = view
            .frame_of
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let (t_buf, t_res, t_ranges, t_verts) = (
            view.terrain.vertex_buffer().is_some(),
            view.terrain.resident_chunks(),
            view.terrain.draw_ranges().len(),
            view.terrain.used_verts(),
        );
        let (g_total, g_chunks) = view.grass.diag_counts();
        let (tn, sn, qun, qn) = view.morph.diag_plan();
        eprintln!(
            "[view3d diag] terrain(buf={t_buf} res={t_res} ranges={t_ranges} verts={t_verts}) \
             grass(total={g_total} chunks={g_chunks}) morph(t={tn} s={sn} q={qun} b={qn}) \
             frame={}",
            frame.is_some()
        );
    }
}

impl View3D {
    /// Per-frame state-side update (called in `prepare`, before `paint`).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame3D) {
        let fallback = (frame.world.width * 0.5, frame.world.height * 0.5);
        let focus = match frame.cam_mode {
            CameraMode::Orbit { focus } => focus,
            CameraMode::Follow { .. } => frame.follow_target.unwrap_or(fallback),
            CameraMode::Planet => fallback,
        };
        self.terrain
            .set_cull(&frame.cam.view, &frame.cam.proj);
        self.terrain
            .ensure_mesh(device, queue, &frame.world, frame.cam_mode, focus);
        queue.write_buffer(self.terrain.cam_buf(), 0, bytemuck::bytes_of::<CamUniform>(&frame.cam));
        self.morph
            .update_gpu(device, queue, &frame);
        // queen visual footprint -> grass trample field (scaled to her body)
        let queen_trample = self.morph.queen_footprint().map(|(x, z, r, s)| [x, z, r, s]);
        self.grass
            .update_gpu(device, queue, &frame, queen_trample);
        // trees (Agent: trees) — after grass, no dependency on morph
        self.tree
            .update_gpu(device, queue, &frame);
        *self.frame_of.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(frame_snap(frame)));
        diag_tick(self);
    }

    /// Record the actual draws (called in `paint`).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let frame = self
            .frame_of
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some((vbuf, _count)) = self.terrain.vertex_buffer() {
            pass.push_debug_group("alife3d.terrain");
            pass.set_pipeline(self.terrain.pipeline());
            pass.set_bind_group(0, self.terrain.bind_group(), &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            for range in self.terrain.draw_ranges() {
                pass.draw(range.clone(), 0..1);
            }
            pass.pop_debug_group();
        }
        // trees (Agent: trees) — after the terrain pass, BEFORE grass so
        // grass draws in front
        pass.push_debug_group("alife3d.tree");
        self.tree.draw_with(pass, frame.as_ref().map(Arc::as_ref));
        pass.pop_debug_group();
        pass.push_debug_group("alife3d.grass");
        self.grass.draw_with(pass, frame.as_ref().map(Arc::as_ref));
        pass.pop_debug_group();
        pass.push_debug_group("alife3d.morph-orgs");
        self.morph.draw_with(pass, frame.as_ref().map(Arc::as_ref));
        pass.pop_debug_group();
        // selection/hover overlay: one extra instanced draw (<= 3 ring quads)
        pass.push_debug_group("alife3d.markers");
        self.morph.draw_markers(pass);
        pass.pop_debug_group();
    }

    /// UI entry: arm this frame's selection markers. Indices are `Frame3D
    /// orgs` slice indices (exactly what `picking::pick_org` /
    /// `picking::hover_org` return; `id_queen = None` falls back to the
    /// snapshot's `is_queen` flag). Cheap: a 3-Option snapshot under a mutex,
    /// consumed by the next `Morph::update_gpu`.
    pub fn set_selection_markers(
        &self,
        id_sel: Option<u32>,
        id_hover: Option<u32>,
        id_queen: Option<u32>,
    ) {
        crate::scene::markers::set_selection(id_sel, id_hover, id_queen);
    }
}

/// Free-function alias for `View3D::set_selection_markers` (usable before the
/// view is reachable, e.g. straight from the UI's click handler).
pub fn set_selection_markers(
    id_sel: Option<u32>,
    id_hover: Option<u32>,
    id_queen: Option<u32>,
) {
    crate::scene::markers::set_selection(id_sel, id_hover, id_queen);
}

/// strips the Arc'd fields out so the hand-off clone is cheap
fn frame_snap(f: &Frame3D) -> Frame3D {
    Frame3D {
        cam: f.cam,
        world: f.world.clone(),
        cam_mode: f.cam_mode,
        dt: f.dt,
        time: f.time,
        viewport_px: f.viewport_px,
        follow_target: f.follow_target,
        orgs: f.orgs.clone(),
        org_arch: f.org_arch.clone(),
        rain_cover: f.rain_cover,
        cloud_cover: f.cloud_cover,
        flash: f.flash,
    }
}

/// Per-frame state carrier handed to egui by the UI layer.
pub struct View3DCallback {
    pub frame: Arc<Frame3D>,
}

impl CallbackTrait for View3DCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(view) = resources.get_mut::<View3D>() {
            view.update(device, queue, &self.frame);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        if let Some(view) = resources.get::<View3D>() {
            view.draw(pass);
        }
    }
}

/// Registers a persistent [`View3D`] in egui's callback resources. Returns
/// false when the wgpu device is unavailable; the UI then keeps the 2D path.
pub fn install(rs: &egui_wgpu::RenderState) -> bool {
    let Some(view) = View3D::new(rs) else {
        return false;
    };
    rs.renderer.write().callback_resources.insert::<View3D>(view);
    true
}

/// UI entry: hand this frame's snapshot to the renderer (egui paint callback).
pub fn enqueue(ui: &mut egui::Ui, rect: egui::Rect, frame: Frame3D) {
    let shape = egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
        rect,
        View3DCallback { frame: Arc::new(frame) },
    ));
    ui.painter().add(shape);
}
