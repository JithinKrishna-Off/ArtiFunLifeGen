//! wgpu pipeline + egui paint-callback glue (coordinator-owned).

pub use crate::scene::{CameraMode, CamUniform, Frame3D};
use crate::View3D;
use egui::epaint::PaintCallbackInfo;
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use std::sync::Arc;

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
        self.grass
            .update_gpu(device, queue, &frame);
        self.morph
            .update_gpu(device, queue, &frame);
        *self.frame_of.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(frame_snap(frame)));
    }

    /// Record the actual draws (called in `paint`).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let frame = self
            .frame_of
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some((vbuf, _count)) = self.terrain.vertex_buffer() {
            pass.set_pipeline(self.terrain.pipeline());
            pass.set_bind_group(0, self.terrain.bind_group(), &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            for range in self.terrain.draw_ranges() {
                pass.draw(range.clone(), 0..1);
            }
        }
        self.grass.draw_with(pass, frame.as_ref().map(Arc::as_ref));
        self.morph.draw_with(pass, frame.as_ref().map(Arc::as_ref));
    }
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
