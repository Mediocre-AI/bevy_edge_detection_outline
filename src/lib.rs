use bevy::{
    anti_alias::{fxaa::fxaa, smaa::smaa},
    asset::{embedded_asset, load_embedded_asset},
    core_pipeline::{
        Core3d, Core3dSystems, FullscreenShader,
        core_3d::{DEPTH_PREPASS_TEXTURE_SUPPORTED, main_opaque_pass_3d, main_transparent_pass_3d},
        prepass::{DepthPrepass, NormalPrepass, ViewPrepassTextures},
        tonemapping::tonemapping,
    },
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_2d_multisampled, texture_depth_2d,
                texture_depth_2d_multisampled, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        sync_component::{SyncComponent, SyncComponentPlugin},
        sync_world::RenderEntity,
        texture::GpuImage,
        view::{
            ExtractedView, ViewDepthStencilTexture, ViewTarget, ViewUniform, ViewUniformOffset,
            ViewUniforms,
        },
    },
};

/// Normal prepass alpha: who wrote a pixel and whether it takes lines.
///
/// ⚠ The normal prepass texture is `Rgb10a2Unorm` (bevy_core_pipeline
/// `NORMAL_PREPASS_FORMAT`): alpha has TWO BITS, so exactly four codes exist
/// — 0, 1/3, 2/3, 1. 2/3 is unused.
pub mod edge_mask {
    /// No lines on this pixel (terrain surfaces, holdouts, smoke).
    pub const SKIP: f32 = 0.0;
    /// Lines, and the pixel's [`super::line_attrs`] in its normal's low bits —
    /// what a pixel-art prepass writes.
    pub const ATTRIBUTED: f32 = 1.0 / 3.0;
    /// Silhouette and crease lines, no attributes (Bevy's own prepass).
    pub const BOTH: f32 = 1.0;
}

/// Per-pixel line attributes of an [`edge_mask::ATTRIBUTED`] pixel, in the low
/// bits of its normal's channels: each channel keeps its top
/// [`NORMAL_BITS`](line_attrs::NORMAL_BITS) for the normal; red's low bits are
/// the flags below, green's and blue's (green the low half) the model the pixel
/// belongs to (0 = none, else 1..=[`OBJECTS`](line_attrs::OBJECTS)). A reader
/// that ignores them sees each component within 1/128 of its packed value.
pub mod line_attrs {
    /// Bits of each normal channel that hold the normal itself.
    pub const NORMAL_BITS: u32 = 7;
    /// How many models the outline tells apart; two that share an id (one in
    /// this many) meet with no line where their depths agree.
    pub const OBJECTS: u32 = 63;
    /// A skin texel: no crease lines inside the skin, and a section line on the
    /// other side of its border (the face against hair, a helmet or a collar).
    pub const SKIN: u32 = 1;
    /// The flag bits above [`SKIN`] hold the model's selection category: 0 for
    /// none, else 1..=[`CATEGORIES`], whose colour is
    /// `EdgeDetection::selection_colors[category - 1]`. Under the PixelArt
    /// operator a selected model gets a contour of that colour just past its
    /// silhouette, under the others all its edges take it.
    pub const CATEGORY_SHIFT: u32 = 1;
    /// How many selection categories a pixel can carry.
    pub const CATEGORIES: u32 = 3;
}

// ──────────────────────────────────────────────
//  Plugin Setup
// ──────────────────────────────────────────────
#[derive(Default)]
pub struct EdgeDetectionPlugin;

impl Plugin for EdgeDetectionPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "edge_detection_shader.wesl");
        embedded_asset!(app, "perlin_noise.png");

        app.register_type::<EdgeDetection>();
        app.add_plugins((
            SyncComponentPlugin::<EdgeDetection>::default(),
            UniformComponentPlugin::<EdgeDetectionUniform>::default(),
        ));
        // We need to get the render app from the main app
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SpecializedRenderPipelines<EdgeDetectionPipeline>>()
            .add_systems(
                ExtractSchedule,
                EdgeDetectionUniform::extract_edge_detection_settings,
            )
            .add_systems(
                Render,
                (
                    prepare_edge_detection_pipelines.in_set(RenderSystems::Prepare),
                    ensure_depth_texture_binding
                        .in_set(RenderSystems::PrepareViews)
                        .ambiguous_with(RenderSystems::PrepareViews),
                ),
            )
            // On a single-sampled view the outlines belong to the opaque scene: they
            // are drawn between the opaque and transparent main passes, so particles
            // and glass in front of an edge cover it. A multisampled view would
            // resolve its transparent pass over that, so there the outlines go over
            // the tonemapped image, before FXAA/SMAA smooth them.
            .add_systems(
                Core3d,
                (
                    edge_detection::<false>
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d)
                        .in_set(Core3dSystems::MainPass)
                        .in_set(EdgeDetectionPassSet),
                    edge_detection::<true>
                        .after(tonemapping)
                        .before(fxaa)
                        .before(smaa)
                        .in_set(Core3dSystems::PostProcess)
                        .in_set(MultisampledEdgeDetectionPassSet),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<EdgeDetectionPipeline>();
    }
}

/// Add `TEXTURE_BINDING` to the depth-texture usages of every extracted camera
/// with [`EdgeDetection`], before its depth texture is made (as Bevy does for
/// occlusion culling): the pass compares the main-pass depth with the prepass
/// depth, so it draws no outline over what only the main pass drew in front (an
/// opaque particle). Done each frame on the extracted camera, it holds however
/// the main-world components were added.
fn ensure_depth_texture_binding(mut cameras: Query<&mut Camera3d, With<EdgeDetection>>) {
    for mut camera in &mut cameras {
        let usages = TextureUsages::from_bits_truncate(camera.depth_texture_usages.0)
            | TextureUsages::TEXTURE_BINDING;
        camera.depth_texture_usages = usages.into();
    }
}

/// Public [`SystemSet`] containing the edge-detection render pass of the
/// single-sampled views, between the opaque and transparent main passes of the
/// [`Core3d`] schedule. A pass that draws over the opaque scene's outlines (e.g.
/// stamped details) orders itself after this set and before
/// `main_transparent_pass_3d`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeDetectionPassSet;

/// Public [`SystemSet`] containing the edge-detection pass of the multisampled
/// views, in the post-process stage. Another post-process pass that writes the
/// same `ViewTarget` on the same camera has no inherent order against it —
/// integrators `configure_sets` an explicit before/after relation.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampledEdgeDetectionPassSet;

// The render entity keeps a synced `EdgeDetection`, its derived
// `EdgeDetectionUniform`, and the per-view derived `EdgeDetectionPipelineId` /
// `DynamicUniformIndex`. ALL must be dropped when `EdgeDetection` is removed
// from the camera in the main world: the render pass's `ViewQuery` matches on
// the pipeline-id + uniform-index pair, so leaving those behind keeps the
// pass running with stale state after the effect is disabled.
impl SyncComponent<RenderApp> for EdgeDetection {
    type Target = (
        EdgeDetection,
        EdgeDetectionUniform,
        EdgeDetectionPipelineId,
        DynamicUniformIndex<EdgeDetectionUniform>,
    );
}

// This contains global data used by the render pipeline. This will be created once on startup.
#[derive(Resource)]
pub struct EdgeDetectionPipeline {
    pub shader: Handle<Shader>,
    pub noise_texture: Handle<Image>,
    pub linear_sampler: Sampler,
    pub nonfiltering_sampler: Sampler,
    pub noise_sampler: Sampler,
    pub layout_with_msaa: BindGroupLayoutDescriptor,
    pub layout_without_msaa: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
}

impl EdgeDetectionPipeline {
    pub fn bind_group_layout(&self, multisampled: bool) -> &BindGroupLayoutDescriptor {
        if multisampled {
            &self.layout_with_msaa
        } else {
            &self.layout_without_msaa
        }
    }
}

impl FromWorld for EdgeDetectionPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "edge_detection_shader.wesl");
        let noise_texture = load_embedded_asset!(world, "perlin_noise.png");

        let layout_with_msaa = BindGroupLayoutDescriptor::new(
            "edge_detection: bind_group_layout with msaa",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // color attachment
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // depth prepass
                    texture_depth_2d_multisampled(),
                    // normal prepass
                    texture_2d_multisampled(TextureSampleType::Float { filterable: false }),
                    // filtering sampler for color/normal
                    sampler(SamplerBindingType::Filtering),
                    // non-filtering sampler for depth prepass
                    sampler(SamplerBindingType::NonFiltering),
                    // perlin-noise texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // perlin-noise sampler
                    sampler(SamplerBindingType::Filtering),
                    // view
                    uniform_buffer::<ViewUniform>(true),
                    // The uniform that will control the effect
                    uniform_buffer::<EdgeDetectionUniform>(true),
                    // main-pass depth
                    texture_depth_2d_multisampled(),
                ),
            ),
        );

        let layout_without_msaa = BindGroupLayoutDescriptor::new(
            "edge_detection: bind_group_layout without msaa",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // color attachment
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // depth prepass
                    texture_depth_2d(),
                    // normal prepass
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // texture sampler
                    sampler(SamplerBindingType::Filtering),
                    // for depth
                    sampler(SamplerBindingType::NonFiltering),
                    // perlin-noise texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // perlin-noise sampler
                    sampler(SamplerBindingType::Filtering),
                    // view
                    uniform_buffer::<ViewUniform>(true),
                    // The uniform that will control the effect
                    uniform_buffer::<EdgeDetectionUniform>(true),
                    // main-pass depth
                    texture_depth_2d(),
                ),
            ),
        );

        let render_device = world.resource::<RenderDevice>();

        let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("edge detection linear sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });
        let nonfiltering_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("edge detection nonfiltering sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..default()
        });

        let noise_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("edge detection noise sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            ..default()
        });

        Self {
            shader,
            noise_texture,
            linear_sampler,
            nonfiltering_sampler,
            noise_sampler,
            layout_with_msaa,
            layout_without_msaa,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
        }
    }
}

impl SpecializedRenderPipeline for EdgeDetectionPipeline {
    type Key = EdgeDetectionKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let targets = vec![Some(ColorTargetState {
            format: key.target_format,
            blend: None,
            write_mask: ColorWrites::ALL,
        })];

        let mut shader_defs = vec![];

        if key.enable_depth {
            shader_defs.push("ENABLE_DEPTH".into());
        }

        if key.enable_normal {
            shader_defs.push("ENABLE_NORMAL".into());
        }

        if key.enable_color {
            shader_defs.push("ENABLE_COLOR".into());
        }

        match key.operator {
            EdgeOperator::Sobel => shader_defs.push("OPERATOR_SOBEL".into()),
            EdgeOperator::RobertsCross => shader_defs.push("OPERATOR_ROBERTS_CROSS".into()),
            EdgeOperator::PixelArt => shader_defs.push("OPERATOR_PIXEL_ART".into()),
        }

        if key.multisampled {
            shader_defs.push("MULTISAMPLED".into());
        }

        match key.projection {
            ProjectionType::Perspective => shader_defs.push("VIEW_PROJECTION_PERSPECTIVE".into()),
            ProjectionType::Orthographic => shader_defs.push("VIEW_PROJECTION_ORTHOGRAPHIC".into()),
            _ => (),
        };

        RenderPipelineDescriptor {
            label: Some("edge_detection: pipeline".into()),
            layout: vec![self.bind_group_layout(key.multisampled).clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("fragment".into()),
                targets,
                ..default()
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Component, Clone, Copy)]
pub struct EdgeDetectionPipelineId(CachedRenderPipelineId);

pub fn prepare_edge_detection_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<EdgeDetectionPipeline>>,
    edge_detection_pipeline: Res<EdgeDetectionPipeline>,
    query: Query<(
        Entity,
        &EdgeDetection,
        Option<&Projection>,
        &ExtractedView,
        &ViewTarget,
    )>,
) {
    for (entity, edge_detection, projection, view, view_target) in &query {
        let multisampled = view_target.sampled_main_texture_view().is_some();
        let id = pipelines.specialize(
            &pipeline_cache,
            &edge_detection_pipeline,
            EdgeDetectionKey::new(edge_detection, view.target_format, multisampled, projection),
        );
        commands.entity(entity).insert(EdgeDetectionPipelineId(id));
    }
}

// ──────────────────────────────────────────────
//  Core structs and types
// ──────────────────────────────────────────────

/// Edge detection operator type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Reflect)]
pub enum EdgeOperator {
    /// 3x3 Sobel filter — 8 samples per type, wider edges, stronger gradients.
    Sobel,
    /// 2x2 Roberts Cross — 4 samples per type, clean 1px edges.
    #[default]
    RobertsCross,
    /// UDLR 4-direction pairwise comparison with silhouette/crease priority.
    /// Guarantees exactly 1px edges. Based on Red Giraffe technique.
    PixelArt,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionType {
    None,
    Perspective,
    Orthographic,
}

impl From<Option<&Projection>> for ProjectionType {
    fn from(proj: Option<&Projection>) -> Self {
        if let Some(projection) = proj {
            return match projection {
                Projection::Perspective(_) => Self::Perspective,
                Projection::Orthographic(_) => Self::Orthographic,
                Projection::Custom(_) => Self::None,
            };
        };

        Self::None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeDetectionKey {
    /// Whether to enable depth-based edge detection.
    /// If `true`, edges will be detected based on depth variations.
    pub enable_depth: bool,
    /// Whether to enable normal-based edge detection.
    /// If `true`, edges will be detected based on normal direction variations.
    pub enable_normal: bool,
    /// Whether to enable color-based edge detection.
    /// If `true`, edges will be detected based on color variations.
    pub enable_color: bool,
    /// Edge detection operator.
    pub operator: EdgeOperator,

    /// The texture format this view renders to (sourced from `ExtractedView::target_format`).
    pub target_format: TextureFormat,
    /// Whether the render target is multisampled.
    pub multisampled: bool,
    /// The projection type of view
    pub projection: ProjectionType,
}

impl EdgeDetectionKey {
    pub fn new(
        edge_detection: &EdgeDetection,
        target_format: TextureFormat,
        multisampled: bool,
        projection: Option<&Projection>,
    ) -> Self {
        Self {
            enable_depth: edge_detection.enable_depth,
            enable_normal: edge_detection.enable_normal,
            enable_color: edge_detection.enable_color,
            operator: edge_detection.operator,

            target_format,
            multisampled,
            projection: projection.into(),
        }
    }
}
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component, Default)]
#[require(DepthPrepass, NormalPrepass)]
pub struct EdgeDetection {
    /// Depth threshold, used to detect edges with significant depth changes.
    /// Areas where the depth variation exceeds this threshold will be marked as edges.
    pub depth_threshold: f32,
    /// Normal threshold, used to detect edges with significant normal direction changes.
    /// Areas where the normal direction variation exceeds this threshold will be marked as edges.
    pub normal_threshold: f32,
    /// Color threshold, used to detect edges with significant color changes.
    /// Areas where the color variation exceeds this threshold will be marked as edges.
    pub color_threshold: f32,

    /// Thickness of the edges detected based on depth variations.
    /// This value controls the width of the edges drawn when depth-based edge detection is enabled.
    /// Higher values result in thicker edges.
    pub depth_thickness: f32,
    /// Thickness of the edges detected based on normal direction variations.
    /// This value controls the width of the edges drawn when normal-based edge detection is enabled.
    /// Higher values result in thicker edges.
    pub normal_thickness: f32,
    /// Thickness of the edges detected based on color variations.
    /// This value controls the width of the edges drawn when color-based edge detection is enabled.
    /// Higher values result in thicker edges.
    pub color_thickness: f32,

    /// Steep angle threshold, used to adjust the depth threshold when viewing surfaces at steep angles.
    /// When the angle between the view direction and the surface normal is very steep, the depth gradient
    /// can appear artificially large, causing non-edge regions to be mistakenly detected as edges.
    /// This threshold defines the angle at which the depth threshold adjustment begins to take effect.
    ///
    /// Range: [0.0, 1.0]
    pub steep_angle_threshold: f32,
    /// Multiplier applied to the depth threshold when the view angle is steep.
    /// When the angle between the view direction and the surface normal exceeds the `steep_angle_threshold`,
    /// the depth threshold is scaled by this multiplier to reduce the likelihood of false edge detection.
    ///
    /// A value of 1.0 means no adjustment, while values greater than 1.0 increase the depth threshold,
    /// making edge detection less sensitive in steep angles.
    ///
    /// Range: [0.0, inf)
    pub steep_angle_multiplier: f32,

    /// Frequency of UV distortion applied to the edge detection process.
    /// This controls how often the distortion effect repeats across the UV coordinates.
    /// Higher values result in more frequent distortion patterns.
    pub uv_distortion_frequency: Vec2,

    /// Strength of UV distortion applied to the edge detection process.
    /// This controls the intensity of the distortion effect.
    /// Higher values result in more pronounced distortion.
    /// Defaults to zero so edges align with the source image; zero frequency alone does not disable distortion.
    pub uv_distortion_strength: Vec2,

    /// Edge color, used to draw the detected edges.
    /// Typically a high-contrast color (e.g., red or black) to visually highlight the edges.
    pub edge_color: Color,

    /// Separate color for silhouette (depth) edges. `None` inherits `edge_color`.
    pub silhouette_color: Option<Color>,
    /// Separate color for crease (normal) edges. `None` inherits `edge_color`.
    pub crease_color: Option<Color>,

    /// The colour of each selection category a pixel can carry
    /// ([`line_attrs::CATEGORY_SHIFT`]): category `k` takes `selection_colors[k - 1]`.
    /// A per-model outline that needs no extra pass — under the PixelArt
    /// operator a contour just past the model's silhouette, all its edges
    /// under the others.
    pub selection_colors: [Color; 3],
    /// How much the Sobel and RobertsCross operators widen a selected model's
    /// edges. The PixelArt operator's selection outline is always one pixel,
    /// just outside the model's one-pixel silhouette.
    pub selection_thickness: f32,

    /// Bright inlines (PixelArt operator): where a model's surface folds
    /// outward (a ridge, the edge of an armour plate) the brighter of the two
    /// faces takes a line of its own colour times `1 + inline_brightness`
    /// (kept below 1), in place of the dark crease. 0 = every crease stays
    /// dark. A fold inward keeps its dark crease.
    pub inline_brightness: f32,

    /// Whether to enable depth-based edge detection.
    /// If `true`, edges will be detected based on depth variations.
    pub enable_depth: bool,
    /// Whether to enable normal-based edge detection.
    /// If `true`, edges will be detected based on normal direction variations.
    pub enable_normal: bool,
    /// Whether to enable color-based edge detection.
    /// If `true`, edges will be detected based on color variations.
    pub enable_color: bool,
    /// Pixel block size.
    pub block_pixel: u32,

    /// Flat surface rejection threshold for the normal Y component.
    /// When all normals in the 3x3 edge detection kernel have Y > this threshold,
    /// the edge is suppressed (treated as a flat surface like hex tiles).
    /// Set to 0.0 to disable flat rejection. Range: [0.0, 1.0]
    pub flat_rejection_threshold: f32,

    /// Edge detection operator: Sobel (3x3, wider) or RobertsCross (2x2, 1px).
    pub operator: EdgeOperator,

    /// How far a silhouette or crease line takes the colour of the pixel it is
    /// drawn on (0..1): at 1 the line is that pixel's colour multiplied by
    /// `tone_color` — a darker shade of the surface itself, the pixel-art
    /// "selective outline". 0 = the flat line colour. Selection outlines keep
    /// their flat colour.
    pub source_tone: f32,
    /// The multiplier a toned line puts on the surface colour.
    pub tone_color: Color,
}

impl Default for EdgeDetection {
    fn default() -> Self {
        Self {
            depth_threshold: 0.9,
            normal_threshold: 0.45,
            color_threshold: 0.1,

            depth_thickness: 1.0,
            normal_thickness: 1.0,
            color_thickness: 1.0,

            steep_angle_threshold: 0.75,
            steep_angle_multiplier: 0.30,

            uv_distortion_frequency: Vec2::splat(0.0),
            uv_distortion_strength: Vec2::ZERO,

            edge_color: Color::BLACK,
            silhouette_color: None,
            crease_color: None,
            selection_colors: [
                Color::srgb(0.95, 0.72, 0.12),
                Color::WHITE,
                Color::srgb(0.9, 0.12, 0.1),
            ],
            selection_thickness: 1.0,
            inline_brightness: 0.0,

            enable_depth: true,
            enable_normal: true,
            enable_color: false,

            block_pixel: 1,

            flat_rejection_threshold: 0.0,

            operator: EdgeOperator::default(),

            source_tone: 0.0,
            tone_color: Color::srgb(0.5, 0.45, 0.6),
        }
    }
}

#[derive(Component, Clone, Copy, ShaderType)]
pub struct EdgeDetectionUniform {
    pub depth_threshold: f32,
    pub normal_threshold: f32,
    pub color_threshold: f32,

    pub depth_thickness: f32,
    pub normal_thickness: f32,
    pub color_thickness: f32,

    pub steep_angle_threshold: f32,
    pub steep_angle_multiplier: f32,

    pub uv_distortion: Vec4,

    pub edge_color: LinearRgba,

    pub silhouette_color: LinearRgba,
    pub crease_color: LinearRgba,
    pub selection_colors: [Vec4; 3],
    pub selection_thickness: f32,
    pub inline_brightness: f32,

    pub block_pixel: u32,
    pub flat_rejection_threshold: f32,
    pub source_tone: f32,
    pub tone_color: LinearRgba,
}

impl From<&EdgeDetection> for EdgeDetectionUniform {
    fn from(ed: &EdgeDetection) -> Self {
        let edge_linear: LinearRgba = ed.edge_color.into();
        Self {
            depth_threshold: ed.depth_threshold,
            normal_threshold: ed.normal_threshold,
            color_threshold: ed.color_threshold,

            depth_thickness: ed.depth_thickness,
            normal_thickness: ed.normal_thickness,
            color_thickness: ed.color_thickness,

            steep_angle_threshold: ed.steep_angle_threshold,
            steep_angle_multiplier: ed.steep_angle_multiplier,

            uv_distortion: Vec4::new(
                ed.uv_distortion_frequency.x,
                ed.uv_distortion_frequency.y,
                ed.uv_distortion_strength.x,
                ed.uv_distortion_strength.y,
            ),

            edge_color: edge_linear,
            silhouette_color: ed
                .silhouette_color
                .map(|c| c.into())
                .unwrap_or(edge_linear),
            crease_color: ed.crease_color.map(|c| c.into()).unwrap_or(edge_linear),
            selection_colors: ed.selection_colors.map(|c| LinearRgba::from(c).to_vec4()),
            selection_thickness: ed.selection_thickness,
            inline_brightness: ed.inline_brightness.max(0.0),

            block_pixel: ed.block_pixel,
            flat_rejection_threshold: ed.flat_rejection_threshold,
            source_tone: ed.source_tone.clamp(0.0, 1.0),
            tone_color: ed.tone_color.into(),
        }
    }
}

impl EdgeDetectionUniform {
    pub fn extract_edge_detection_settings(
        mut commands: Commands,
        mut query: Extract<Query<(RenderEntity, &EdgeDetection)>>,
    ) {
        if !DEPTH_PREPASS_TEXTURE_SUPPORTED {
            info_once!(
                "Disable edge detection on this platform because depth textures aren't supported correctly"
            );
            return;
        }

        for (entity, edge_detection) in query.iter_mut() {
            let mut entity_commands = commands
                .get_entity(entity)
                .expect("Edge Detection entity wasn't synced.");

            entity_commands.insert((*edge_detection, EdgeDetectionUniform::from(edge_detection)));
        }
    }
}

// ──────────────────────────────────────────────
//  Render pass (a system in the `Core3d` schedule as of Bevy 0.19)
// ──────────────────────────────────────────────

/// Runs the edge-detection pass for the current view, when its sampling matches
/// `MULTISAMPLED` (the two placements are registered in [`EdgeDetectionPlugin`]).
///
/// [`ViewQuery`] fetches the components of the view currently being rendered; if
/// the view has no [`EdgeDetectionPipelineId`] / [`DynamicUniformIndex`] (i.e. no
/// [`EdgeDetection`] camera component) the parameter fails validation and the
/// system is skipped for that view — matching the old `ViewNode` behaviour.
pub fn edge_detection<const MULTISAMPLED: bool>(
    view: ViewQuery<(
        &Msaa,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewDepthStencilTexture,
        &ViewUniformOffset,
        &DynamicUniformIndex<EdgeDetectionUniform>,
        &EdgeDetectionPipelineId,
    )>,
    edge_detection_pipeline: Res<EdgeDetectionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    view_uniforms: Res<ViewUniforms>,
    ed_uniforms: Res<ComponentUniforms<EdgeDetectionUniform>>,
    mut ctx: RenderContext,
) {
    let (
        msaa,
        view_target,
        prepass_textures,
        scene_depth,
        view_uniform_index,
        ed_uniform_index,
        edge_detection_pipeline_id,
    ) = view.into_inner();
    let multisampled = *msaa != Msaa::Off;
    if multisampled != MULTISAMPLED {
        return;
    }

    let Some(pipeline) = pipeline_cache.get_render_pipeline(edge_detection_pipeline_id.0) else {
        info!("pipeline not found");
        return;
    };

    let (Some(depth_texture), Some(normal_texture)) =
        (&prepass_textures.depth, &prepass_textures.normal)
    else {
        info!("depth or normal texture not found");
        return;
    };
    let Some(scene_depth) = scene_depth.attachment.depth_stencil_views().depth_only_view() else {
        info!("main-pass depth view not found");
        return;
    };

    let Some(noise_texture) = gpu_images.get(&edge_detection_pipeline.noise_texture) else {
        info!("noise texture not found");
        return;
    };

    let Some(view_uniforms_binding) = view_uniforms.uniforms.binding() else {
        info!("view uniforms not found");
        return;
    };

    let Some(ed_uniform_binding) = ed_uniforms.uniforms().binding() else {
        info!("edge detection uniform not found");
        return;
    };

    // This will start a new "post process write", obtaining two texture
    // views from the view target - a `source` and a `destination`.
    // `source` is the "current" main texture and you _must_ write into
    // `destination` because calling `post_process_write()` on the
    // [`ViewTarget`] will internally flip the [`ViewTarget`]'s main
    // texture to the `destination` texture. Failing to do so will cause
    // the current main texture information to be lost.
    let post_process = view_target.post_process_write();

    // The bind_group gets created each frame.
    //
    // Normally, you would create a bind_group in the Queue set,
    // but this doesn't work with the post_process_write().
    // The reason it doesn't work is because each post_process_write will alternate the source/destination.
    // The only way to have the correct source/destination for the bind_group
    // is to make sure you get it during the pass execution.
    let bind_group = ctx.render_device().create_bind_group(
        "edge_detection_bind_group",
        &pipeline_cache
            .get_bind_group_layout(edge_detection_pipeline.bind_group_layout(multisampled)),
        // It's important for this to match the BindGroupLayout defined in the PostProcessPipeline
        &BindGroupEntries::sequential((
            // Make sure to use the source view
            post_process.source,
            // Use depth prepass
            &depth_texture.texture.default_view,
            // Use normal prepass
            &normal_texture.texture.default_view,
            // Use simple texture sampler
            &edge_detection_pipeline.linear_sampler,
            // nonfiltering sampler for depth
            &edge_detection_pipeline.nonfiltering_sampler,
            // Use noise texture
            &noise_texture.texture_view,
            // Use noise texture sampler
            &edge_detection_pipeline.noise_sampler,
            // view uniform binding
            view_uniforms_binding,
            // Set the uniform binding
            ed_uniform_binding,
            // main-pass depth
            scene_depth,
        )),
    );

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("edge_detection_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: post_process.destination,
            depth_slice: None,
            resolve_target: None,
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(
        0,
        &bind_group,
        &[view_uniform_index.offset, ed_uniform_index.index()],
    );
    render_pass.draw(0..3, 0..1);
}
