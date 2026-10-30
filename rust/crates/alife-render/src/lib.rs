//! World renderer for Artificial Life: baked terrain textures, camera,
//! entity painting, weather overlays and minimap. Pure egui painter, no GPU
//! backend required.

pub mod brush;
pub mod camera;
pub mod paint;
pub mod terrain;
pub mod trails;
pub mod weather;

pub use brush::{brush_color, draw_brush_preview, BrushKind};
pub use camera::{CanvasResponse, DynFields, ViewMode, WorldView};
pub use paint::{draw_minimap, draw_world, DrawConfig};
pub use weather::{draw_weather, weather_layer, WeatherLayer};
