//! Edge Detection Shader
//!
//! This shader implements edge detection based on depth, normal, and color gradients.
//! Three operators are supported via shader defs:
//!   - OPERATOR_SOBEL:        3x3 Sobel filter — 8 samples per type, wider edges, stronger gradients.
//!   - OPERATOR_ROBERTS_CROSS: 2x2 Roberts Cross — 4 samples per type, clean 1px edges.
//!   - OPERATOR_PIXEL_ART:    UDLR 4-direction pairwise comparison — 1px guaranteed, silhouettes drawn
//!                            just outside the nearer surface (alpha 2 marks them for a compositor),
//!                            per-pixel line attributes from the normal prepass.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_render::view::View;
import bevy_pbr::render::view_transformations::uv_to_ndc;

@group(0) @binding(0) var screen_texture: texture_2d<f32>;

@if(MULTISAMPLED)
@group(0) @binding(1) var depth_prepass_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(1) var depth_prepass_texture: texture_depth_2d;

@if(MULTISAMPLED)
@group(0) @binding(2) var normal_prepass_texture: texture_multisampled_2d<f32>;
@else
@group(0) @binding(2) var normal_prepass_texture: texture_2d<f32>;

@group(0) @binding(3) var filtering_sampler: sampler;
@group(0) @binding(4) var depth_sampler: sampler;

@group(0) @binding(5) var noise_texture: texture_2d<f32>;
@group(0) @binding(6) var noise_sampler: sampler;

@group(0) @binding(7) var<uniform> view: View;
@group(0) @binding(8) var<uniform> ed_uniform: EdgeDetectionUniform;

// The main pass's depth so far: the prepass's surfaces and what only the main
// pass draws (opaque particles).
@if(MULTISAMPLED)
@group(0) @binding(9) var scene_depth_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(9) var scene_depth_texture: texture_depth_2d;

struct EdgeDetectionUniform {
    depth_threshold: f32,
    normal_threshold: f32,
    color_threshold: f32,

    depth_thickness: f32,
    normal_thickness: f32,
    color_thickness: f32,

    steep_angle_threshold: f32,
    steep_angle_multiplier: f32,

    // xy: distortion frequency; zw: distortion strength
    uv_distortion: vec4f,

    edge_color: vec4f,

    silhouette_color: vec4f,
    crease_color: vec4f,
    highlight_color: vec4f,
    highlight_thickness: f32,

    block_pixel: u32,
    flat_rejection_threshold: f32,
    source_tone: f32,
    tone_color: vec4f,
}

// -----------------------
// View Transformation ---
// -----------------------

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near() -> f32 {
    return view.clip_from_view[3][2];
}

/// Convert ndc depth to linear view z. 
/// Note: Depth values in front of the camera will be negative as -z is forward
fn depth_ndc_to_view_z(ndc_depth: f32) -> f32 {
@if(VIEW_PROJECTION_PERSPECTIVE) {
    return -perspective_camera_near() / ndc_depth;
} @elif(VIEW_PROJECTION_ORTHOGRAPHIC) {
    return -(view.clip_from_view[3][2] - ndc_depth) / view.clip_from_view[2][2];
} @else {
    let view_pos = view.view_from_clip * vec4f(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
}
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = view.world_from_clip * vec4f(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn calculate_view(world_position: vec3f) -> vec3f {
@if(VIEW_PROJECTION_ORTHOGRAPHIC) {
        // Orthographic view vector
        return normalize(vec3f(view.clip_from_world[0].z, view.clip_from_world[1].z, view.clip_from_world[2].z));
} @else {
        // Only valid for a perspective projection
        return normalize(view.world_position.xyz - world_position.xyz);
}
}

// -----------------------
// Depth Detection -------
// -----------------------

fn prepass_depth(uv: vec2f) -> f32 {
@if(MULTISAMPLED) {
    let pixel_coord = vec2i(uv * texture_size);
    let depth = textureLoad(depth_prepass_texture, pixel_coord, sample_index_i);
} @else {
    let depth = textureSample(depth_prepass_texture, depth_sampler, uv);
}
    return depth;
}

fn scene_depth_at(px: vec2i) -> f32 {
@if(MULTISAMPLED) {
    let depth = textureLoad(scene_depth_texture, px, sample_index_i);
} @else {
    let depth = textureLoad(scene_depth_texture, px, 0);
}
    return depth;
}

fn prepass_depth_at(px: vec2i) -> f32 {
@if(MULTISAMPLED) {
    let depth = textureLoad(depth_prepass_texture, px, sample_index_i);
} @else {
    let depth = textureLoad(depth_prepass_texture, px, 0);
}
    return depth;
}

// Something the prepass never drew (an opaque particle) lies in front of every
// prepass surface whose edge could land here (Bevy's reverse Z grows toward the
// camera). The outlines belong to the prepass's surfaces, so none is drawn over
// it. One that stands behind a surface's edge (a flame behind a hill) leaves
// that surface's outline drawn.
fn covered_by_main_pass_only(uv: vec2f) -> bool {
    let last = vec2i(texture_size) - vec2i(1);
    let px = clamp(vec2i(uv * texture_size), vec2i(0), last);
    let scene = scene_depth_at(px);
    if (scene <= prepass_depth_at(px) + 1e-6) {
        return false;
    }
    // As far as any edge test probes (a hover contour reaches further).
    let thickness = max(ed_uniform.depth_thickness, ed_uniform.normal_thickness) * max(ed_uniform.highlight_thickness, 1.0);
    let reach = max(i32(ceil(thickness)), 1);
    var nearest = 0.0;
    for (var y = -reach; y <= reach; y++) {
        for (var x = -reach; x <= reach; x++) {
            nearest = max(nearest, prepass_depth_at(clamp(px + vec2i(x, y), vec2i(0), last)));
        }
    }
    return scene > nearest + 1e-6;
}

fn prepass_view_z(uv: vec2f) -> f32 {
    let depth = prepass_depth(uv);
    return depth_ndc_to_view_z(depth);
}

fn detect_edge_depth(uv: vec2f, thickness: f32, fresnel: f32) -> f32 {
    let offset = texel_size * thickness;

@if(OPERATOR_SOBEL) {
    // 3x3 Sobel: horizontal/vertical gradient from 8 neighbors
    let d_tl = prepass_view_z(uv + vec2f(-offset.x,  offset.y));
    let d_t  = prepass_view_z(uv + vec2f(      0.0,  offset.y));
    let d_tr = prepass_view_z(uv + vec2f( offset.x,  offset.y));
    let d_l  = prepass_view_z(uv + vec2f(-offset.x,       0.0));
    let d_r  = prepass_view_z(uv + vec2f( offset.x,       0.0));
    let d_bl = prepass_view_z(uv + vec2f(-offset.x, -offset.y));
    let d_b  = prepass_view_z(uv + vec2f(      0.0, -offset.y));
    let d_br = prepass_view_z(uv + vec2f( offset.x, -offset.y));

    let gx = -d_tl - 2.0*d_l - d_bl + d_tr + 2.0*d_r + d_br;
    let gy = -d_tl - 2.0*d_t - d_tr + d_bl + 2.0*d_b + d_br;
    let grad = max(abs(gx), abs(gy));
    let view_z = abs(prepass_view_z(uv));
} @else {
    // 2x2 Roberts Cross: diagonal differences from 4 samples
    let d00 = prepass_view_z(uv);
    let d10 = prepass_view_z(uv + vec2f(offset.x, 0.0));
    let d01 = prepass_view_z(uv + vec2f(0.0, offset.y));
    let d11 = prepass_view_z(uv + offset);

    let diff_diag0 = d00 - d11;
    let diff_diag1 = d10 - d01;
    let grad = max(abs(diff_diag0), abs(diff_diag1));
    let view_z = abs(d00);
}

    let steep_angle_adjustment =
        smoothstep(ed_uniform.steep_angle_threshold, 1.0, fresnel) * ed_uniform.steep_angle_multiplier * view_z;

    return f32(grad > ed_uniform.depth_threshold * (1.0 + steep_angle_adjustment));
}

// -----------------------
// Normal Detection ------
// -----------------------

fn prepass_normal_unpack(uv: vec2f) -> vec3f {
    let normal_packed = prepass_normal(uv);
    return normalize(normal_packed.xyz * 2.0 - vec3f(1.0));
}

fn prepass_normal(uv: vec2f) -> vec3f {
    return packed_normal(prepass_normal_raw(uv));
}

/// The packed normal of a raw prepass texel: an ATTRIBUTED pixel's red channel
/// keeps only its top bits for the normal (see Line Attributes).
fn packed_normal(raw: vec4f) -> vec3f {
    if (raw.a >= 0.1667 && raw.a < 0.5) {
        let normal = vec3<u32>(round(raw.rgb * 1023.0)) >> vec3<u32>(ATTR_LOW_BITS);
        return vec3f(normal) / f32((1u << (10u - ATTR_LOW_BITS)) - 1u);
    }
    return raw.xyz;
}

// Loaded, not filtered: the low bits carry line attributes.
fn prepass_normal_raw(uv: vec2f) -> vec4f {
    let pixel_coord = clamp(vec2i(uv * texture_size), vec2i(0), vec2i(texture_size) - vec2i(1));
@if(MULTISAMPLED) {
    let normal = textureLoad(normal_prepass_texture, pixel_coord, sample_index_i);
} @else {
    let normal = textureLoad(normal_prepass_texture, pixel_coord, 0);
}
    return normal;
}

fn detect_edge_normal(uv: vec2f, thickness: f32) -> f32 {
    let offset = texel_size * thickness;

@if(OPERATOR_SOBEL) {
    let n_tl = prepass_normal(uv + vec2f(-offset.x,  offset.y));
    let n_t  = prepass_normal(uv + vec2f(      0.0,  offset.y));
    let n_tr = prepass_normal(uv + vec2f( offset.x,  offset.y));
    let n_l  = prepass_normal(uv + vec2f(-offset.x,       0.0));
    let n_r  = prepass_normal(uv + vec2f( offset.x,       0.0));
    let n_bl = prepass_normal(uv + vec2f(-offset.x, -offset.y));
    let n_b  = prepass_normal(uv + vec2f(      0.0, -offset.y));
    let n_br = prepass_normal(uv + vec2f( offset.x, -offset.y));

    let gx = -n_tl - 2.0*n_l - n_bl + n_tr + 2.0*n_r + n_br;
    let gy = -n_tl - 2.0*n_t - n_tr + n_bl + 2.0*n_b + n_br;
    let grad = sqrt(dot(gx, gx) + dot(gy, gy));
} @else {
    let n00 = prepass_normal(uv);
    let n10 = prepass_normal(uv + vec2f(offset.x, 0.0));
    let n01 = prepass_normal(uv + vec2f(0.0, offset.y));
    let n11 = prepass_normal(uv + offset);

    let diff0 = n00 - n11;
    let diff1 = n10 - n01;
    let grad = sqrt(dot(diff0, diff0) + dot(diff1, diff1));
}

    return f32(grad > ed_uniform.normal_threshold);
}

// ----------------------
// Color Detection ------
// ----------------------

fn prepass_color(uv: vec2f) -> vec3f {
    return textureSample(screen_texture, filtering_sampler, uv).rgb;
}

fn detect_edge_color(uv: vec2f, thickness: f32) -> f32 {
    let offset = texel_size * thickness;

@if(OPERATOR_SOBEL) {
    let c_tl = prepass_color(uv + vec2f(-offset.x,  offset.y));
    let c_t  = prepass_color(uv + vec2f(      0.0,  offset.y));
    let c_tr = prepass_color(uv + vec2f( offset.x,  offset.y));
    let c_l  = prepass_color(uv + vec2f(-offset.x,       0.0));
    let c_r  = prepass_color(uv + vec2f( offset.x,       0.0));
    let c_bl = prepass_color(uv + vec2f(-offset.x, -offset.y));
    let c_b  = prepass_color(uv + vec2f(      0.0, -offset.y));
    let c_br = prepass_color(uv + vec2f( offset.x, -offset.y));

    let gx = -c_tl - 2.0*c_l - c_bl + c_tr + 2.0*c_r + c_br;
    let gy = -c_tl - 2.0*c_t - c_tr + c_bl + 2.0*c_b + c_br;
    let grad = sqrt(dot(gx, gx) + dot(gy, gy));
} @else {
    let c00 = prepass_color(uv);
    let c10 = prepass_color(uv + vec2f(offset.x, 0.0));
    let c01 = prepass_color(uv + vec2f(0.0, offset.y));
    let c11 = prepass_color(uv + offset);

    let diff0 = c00 - c11;
    let diff1 = c10 - c01;
    let grad = sqrt(dot(diff0, diff0) + dot(diff1, diff1));
}

    return f32(grad > ed_uniform.color_threshold);
}

// -----------------------
// Line Attributes -------
// -----------------------

// The normal prepass alpha (Rgb10a2Unorm: two bits) says who wrote a pixel
// (`edge_mask`): 0 SKIP, no lines · 1/3 ATTRIBUTED, a pixel-art prepass: lines,
// and in each normal channel's low bits (`line_attrs`) the flags below (red)
// and the model the pixel belongs to (green low half, blue high half; 0 none)
// · 1 BOTH, Bevy's own prepass: lines. 2/3 is unused. Thresholds sit between
// the codes.
const ATTR_LOW_BITS: u32 = 3u; // 10 - line_attrs::NORMAL_BITS
const ATTR_SKIN: u32 = 1u;
const ATTR_HIGHLIGHT: u32 = 2u;

struct LineAttrs {
    lines: bool,
    skin: bool,
    highlight: bool,
    object: u32,
}

fn line_attrs(raw: vec4f) -> LineAttrs {
    var attrs = LineAttrs(raw.a >= 0.1667, false, false, 0u);
    if (attrs.lines && raw.a < 0.5) {
        let mask = (1u << ATTR_LOW_BITS) - 1u;
        let bits = vec3<u32>(round(raw.rgb * 1023.0)) & vec3<u32>(mask);
        attrs.skin = (bits.r & ATTR_SKIN) != 0u;
        attrs.highlight = (bits.r & ATTR_HIGHLIGHT) != 0u;
        attrs.object = bits.g | (bits.b << ATTR_LOW_BITS);
    }
    return attrs;
}

fn line_attrs_at(uv: vec2f) -> LineAttrs {
    return line_attrs(prepass_normal_raw(uv));
}

/// The screen value above which a pixel counts as an emitter (romance #907).
const EMITTER_LEVEL: f32 = 1.6;

// -----------------------
// PixelArt Operator ------
// -----------------------

/// UDLR pairwise normal comparison for crease detection.
/// Directional tie-breaking: only marks one side to ensure 1px crease edges.
/// If the neighbor is SKIP (alpha < 0.1667), center always wins — otherwise
/// the edge would be lost because the SKIP side never draws edges.
@if(OPERATOR_PIXEL_ART)
fn check_crease(uv: vec2f, offset: vec2f) -> bool {
    let n_center = prepass_normal(uv);
    let raw_neighbor = prepass_normal_raw(uv + offset);
    let n_neighbor = packed_normal(raw_neighbor);
    let diff = n_center - n_neighbor;
    if (dot(diff, diff) <= ed_uniform.normal_threshold * ed_uniform.normal_threshold) {
        return false;
    }
    // If neighbor is SKIP, center always wins (SKIP side won't draw any edge).
    if (raw_neighbor.a < 0.1667) {
        return true;
    }
    // Tie-break: only mark on the pixel with larger component sum.
    let sum_c = n_center.x + n_center.y + n_center.z;
    let sum_n = n_neighbor.x + n_neighbor.y + n_neighbor.z;
    return sum_c >= sum_n;
}

fn pixelate_uv(uv: vec2f, dims: vec2f, block_px: f32) -> vec2f {
    let b = max(block_px, 1.0);
    let cell = floor(uv * dims / b);
    let center = (cell * b + 0.5 * b) / dims; // sample at block center
    return center;
}

var<private> texture_size: vec2f;
var<private> texel_size: vec2f;
var<private> sample_index_i: i32 = 0;

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    in: FullscreenVertexOutput
) -> @location(0) vec4f {
@if(MULTISAMPLED) {
    sample_index_i = i32(sample_index);
}

    texture_size = vec2f(textureDimensions(screen_texture, 0));
    texel_size = 1.0 / texture_size;

    let near_ndc_pos = vec3f(uv_to_ndc(in.uv), 1.0);
    let near_world_pos = position_ndc_to_world(near_ndc_pos);

    let view_direction = calculate_view(near_world_pos);
    
    let normal = prepass_normal_unpack(in.uv);
    let fresnel = 1.0 - saturate(dot(normal, view_direction));

    let sample_uv = in.position.xy * min(texel_size.x, texel_size.y);
    let noise = textureSample(noise_texture, noise_sampler, sample_uv * ed_uniform.uv_distortion.xy);

    let uv_noise = in.uv + noise.xy * ed_uniform.uv_distortion.zw;
    let block_pixel = max(f32(ed_uniform.block_pixel), 1.0);
    let uv_noise_px = pixelate_uv(uv_noise, texture_size, f32(block_pixel));
    let uv_px = pixelate_uv(in.uv, texture_size, f32(block_pixel));

    var edge = 0.0;
    var edge_from_color = false;
    // The pixel whose colour a toned line takes: the center, or for a
    // silhouette the nearer neighbour whose outline lies here.
    var line_owner_uv = uv_px;
    var outside_line = false;
    var resolved_edge_color = ed_uniform.edge_color;
    // A HIGHLIGHT entity's outer contour: its pixels that touch anything that
    // is not it (PixelArt operator).
    var at_highlight_contour = false;
    // Line attributes of the CENTER pixel. Creases tie-break within one object
    // and section lines lie on the non-skin side, so the center owns them; a
    // PixelArt silhouette belongs to the nearer neighbour (`line_owner_uv`).
    let center_attrs = line_attrs_at(uv_noise_px);
    let center_is_highlight = center_attrs.highlight;

@if(OPERATOR_PIXEL_ART) {
    // PixelArt operator: UDLR pairwise comparison with silhouette/crease priority.
    let px_offset = vec2f(block_pixel, block_pixel) / texture_size;

    // A HIGHLIGHT entity's contour reaches this many pixels in; ordinary edges
    // everywhere (its inner ones included) keep the authored one-pixel checks.
    let highlight_scale = select(
        1.0,
        max(ed_uniform.highlight_thickness, 1.0),
        center_is_highlight,
    );
    let offsets = array<vec2f, 4>(
        vec2f(0.0, px_offset.y),   // Down (+y in UV space)
        vec2f(0.0, -px_offset.y),  // Up
        vec2f(-px_offset.x, 0.0),  // Left
        vec2f(px_offset.x, 0.0),   // Right
    );

    // A hovered or selected entity (HIGHLIGHT) is outlined only where its pixels
    // meet anything that is not it — one outer contour in highlight_color, as
    // wide as highlight_thickness. Its inner lines stay ordinary edges.
    if (center_is_highlight) {
        let reach = i32(ceil(highlight_scale));
        for (var k = 1; k <= reach; k++) {
            for (var i = 0; i < 4; i++) {
                if (!line_attrs_at(uv_noise_px + offsets[i] * f32(k)).highlight) {
                    at_highlight_contour = true;
                }
            }
        }
    }

    var is_silhouette = false;
    var is_section = false;
    var is_crease = false;
    var at_depth_boundary = false;

    // 1) Silhouette — drawn just OUTSIDE the nearer surface: the pixel beside
    //    the nearest lined neighbour that stands in front of it by the depth
    //    threshold takes the line in that neighbour's colour (an empty canvas
    //    pixel included), so a small figure keeps its outermost pixels. Either
    //    side of such a boundary draws no crease or section line.
@if(ENABLE_DEPTH) {
    let center_z = prepass_view_z(uv_noise_px);
    var nearest_z = center_z;
    for (var i = 0; i < 4; i++) {
        let neighbor_uv = uv_noise_px + offsets[i];
        if (!line_attrs_at(neighbor_uv).lines) {
            continue;
        }
        let neighbor_z = prepass_view_z(neighbor_uv);
        let neighbor_fresnel = 1.0 - saturate(dot(prepass_normal_unpack(neighbor_uv), view_direction));
        let threshold = ed_uniform.depth_threshold * (1.0
            + smoothstep(ed_uniform.steep_angle_threshold, 1.0, neighbor_fresnel)
                * ed_uniform.steep_angle_multiplier * abs(neighbor_z));
        if (neighbor_z - center_z > threshold) {
            at_depth_boundary = true;
            if (neighbor_z > nearest_z) {
                nearest_z = neighbor_z;
                line_owner_uv = neighbor_uv;
                is_silhouette = true;
            }
        } else if (center_z - neighbor_z > threshold && center_attrs.lines) {
            at_depth_boundary = true;
        }
    }
}

    // 2) Model line — where two models meet with no depth gap (soldiers side
    //    by side, a general before his men at a far zoom), the farther one (on
    //    a tie the lower id) takes the nearer one's silhouette line, as above.
    if (!is_silhouette && center_attrs.object != 0u) {
        let model_z = prepass_view_z(uv_noise_px);
        for (var m = 0; m < 4; m++) {
            let other_uv = uv_noise_px + offsets[m];
            let other = line_attrs_at(other_uv);
            if (other.object == 0u || other.object == center_attrs.object) {
                continue;
            }
            at_depth_boundary = true;
            let other_z = prepass_view_z(other_uv);
            if (other_z > model_z || (other_z == model_z && other.object > center_attrs.object)) {
                is_silhouette = true;
                line_owner_uv = other_uv;
                break;
            }
        }
    }

    // 3) Section line — where skin meets anything else on one model (the face
    //    against hair, a helmet or a collar), the other side takes a
    //    silhouette-strength line, so the small face keeps its pixels.
    if (!is_silhouette && !at_depth_boundary && center_attrs.lines && !center_attrs.skin) {
        for (var s = 0; s < 4; s++) {
            if (line_attrs_at(uv_noise_px + offsets[s]).skin) {
                is_section = true;
                break;
            }
        }
    }

    // 4) Crease search — skip on skin, and if a silhouette or section line was
    //    found OR at a depth boundary. Suppressing crease at depth boundaries
    //    prevents the background pixel from drawing a false crease (different
    //    objects = different normals).
@if(ENABLE_NORMAL) {
    if (!is_silhouette && !is_section && !at_depth_boundary && center_attrs.lines && !center_attrs.skin) {
        for (var j = 0; j < 4; j++) {
            if (check_crease(uv_noise_px, offsets[j])) {
                is_crease = true;
                break;
            }
        }
    }
}

    // 5) Type-specific color
    // A highlighted model's own contour pixel stays its own (gold, at its depth).
    outside_line = is_silhouette && !at_highlight_contour;
    if (is_silhouette || is_section) {
        edge = 1.0;
        resolved_edge_color = ed_uniform.silhouette_color;
    } else if (is_crease) {
        edge = 1.0;
        resolved_edge_color = ed_uniform.crease_color;
    }
    if (at_highlight_contour) {
        edge = 1.0;
    }

@if(ENABLE_COLOR) {
    if (edge < 1.0) {
        let edge_color_val = detect_edge_color(uv_noise_px, ed_uniform.color_thickness);
        if (edge_color_val > 0.0) {
            edge = 1.0;
            edge_from_color = true;
            resolved_edge_color = ed_uniform.edge_color;
        }
    }
}

} @else {
    // Sobel / Roberts Cross operators with silhouette/crease priority.
@if(ENABLE_DEPTH) {
    var edge_depth = detect_edge_depth(uv_noise_px, ed_uniform.depth_thickness, fresnel);
    if (edge_depth <= 0.0 && center_is_highlight && ed_uniform.highlight_thickness > 1.0) {
        edge_depth = detect_edge_depth(
            uv_noise_px,
            ed_uniform.depth_thickness * ed_uniform.highlight_thickness,
            fresnel,
        );
    }
    if (edge_depth > 0.0) {
        edge = 1.0;
        resolved_edge_color = ed_uniform.silhouette_color;
    }
}

@if(ENABLE_NORMAL) {
    if (edge < 1.0) {
        var edge_normal = detect_edge_normal(uv_noise_px, ed_uniform.normal_thickness);
        if (edge_normal <= 0.0 && center_is_highlight && ed_uniform.highlight_thickness > 1.0) {
            edge_normal = detect_edge_normal(
                uv_noise_px,
                ed_uniform.normal_thickness * ed_uniform.highlight_thickness,
            );
        }
        if (edge_normal > 0.0) {
            edge = 1.0;
            resolved_edge_color = ed_uniform.crease_color;
        }
    }
}

@if(ENABLE_COLOR) {
    if (edge < 1.0) {
        var edge_color_val = detect_edge_color(uv_noise_px, ed_uniform.color_thickness);
        if (edge_color_val <= 0.0 && center_is_highlight && ed_uniform.highlight_thickness > 1.0) {
            edge_color_val = detect_edge_color(
                uv_noise_px,
                ed_uniform.color_thickness * ed_uniform.highlight_thickness,
            );
        }
        if (edge_color_val > 0.0) {
            edge = 1.0;
            edge_from_color = true;
            resolved_edge_color = ed_uniform.edge_color;
        }
    }
}
}

    // Highlight accent (`line_attrs::HIGHLIGHT`): the PixelArt operator draws it
    // on the entity's outer contour only; the other operators on all its edges.
    // Ownership: positive-axis kernels (Sobel/RobertsCross always; the color
    // detector in EVERY operator) can detect an object's left/top edge on the
    // NEIGHBOR fragment, so the center sample may belong to the background —
    // probe the producing kernel's footprint then. The PixelArt contour is the
    // highlighted entity's own pixel and skips the probe. Bleed onto a touching
    // object's boundary is fine: that boundary is visually the highlighted
    // object's silhouette.
    if (edge > 0.0) {
        var highlight_owner = center_is_highlight;
        var probe_t = 0.0;
@if(OPERATOR_PIXEL_ART) {
        // The contour alone takes the accent; inner edges keep their own colour.
        highlight_owner = at_highlight_contour;
} @else {
        probe_t = max(max(ed_uniform.depth_thickness, ed_uniform.normal_thickness), 1.0);
@if(ENABLE_COLOR) {
        probe_t = max(probe_t, ed_uniform.color_thickness);
}
}
        if (!highlight_owner && probe_t > 0.0) {
            for (var iy = -1; iy <= 1; iy++) {
                for (var ix = -1; ix <= 1; ix++) {
                    if (line_attrs_at(uv_noise_px + texel_size * vec2f(f32(ix), f32(iy)) * probe_t).highlight) {
                        highlight_owner = true;
                    }
                }
            }
        }
        if (highlight_owner) {
            resolved_edge_color = ed_uniform.highlight_color;
        } else if (ed_uniform.source_tone > 0.0) {
            // Selective outline: the line is a darker shade of the surface it
            // is drawn on (the center sample owns the edge).
            let own = textureSampleLevel(screen_texture, filtering_sampler, line_owner_uv, 0.0).rgb;
            resolved_edge_color = vec4f(
                mix(resolved_edge_color.rgb, own * ed_uniform.tone_color.rgb, ed_uniform.source_tone),
                resolved_edge_color.a,
            );
        }
    }

    // An emitter takes no ink line: a lantern a few pixels wide is all edge, and
    // a line over it hid its glow (romance #907); nor casts one around it — a
    // silhouette is tested at its owner. A hover contour may cross it.
    // Emitters burn at several times their colour; deband dither (~1/255) and a
    // hover's surface tint stay below this bar, so bright surfaces keep their ink.
    if (edge > 0.0 && !at_highlight_contour) {
        let lit = textureSampleLevel(screen_texture, filtering_sampler, line_owner_uv, 0.0).rgb;
        if (max(lit.r, max(lit.g, lit.b)) > EMITTER_LEVEL) {
            edge = 0.0;
        }
    }

    // SKIP pixels take no line from these operators' neighbour kernels either:
    // a pixel whose whole neighbourhood is SKIP (terrain surfaces) is cleared.
    // The PixelArt operator owns every edge at its center pixel and needs no pass.
@if(!OPERATOR_PIXEL_ART) {
    if (edge > 0.0 && !center_attrs.lines) {
        let max_thickness = max(ed_uniform.depth_thickness, ed_uniform.normal_thickness);
        var any_lines = false;
        for (var iy = -1; iy <= 1; iy++) {
            for (var ix = -1; ix <= 1; ix++) {
                let offset_uv = uv_noise_px + texel_size * vec2f(f32(ix), f32(iy)) * max_thickness;
                any_lines = any_lines || line_attrs_at(offset_uv).lines;
            }
        }
        if (!any_lines) {
            edge = 0.0;
        }
    }
}

    // Flat surface rejection (fallback for StandardMaterial entities without custom prepass).
    // A hover contour stays whole: it is drawn without a depth or normal edge.
    if (ed_uniform.flat_rejection_threshold > 0.0 && edge > 0.0 && !at_highlight_contour) {
        let max_thickness = max(ed_uniform.depth_thickness, ed_uniform.normal_thickness);
        let reject_t = ed_uniform.flat_rejection_threshold;
        var min_ny = 1.0;
        for (var iy = -1; iy <= 1; iy++) {
            for (var ix = -1; ix <= 1; ix++) {
                let offset_uv = uv_noise_px + texel_size * vec2f(f32(ix), f32(iy)) * max_thickness;
                let n = prepass_normal_unpack(offset_uv);
                min_ny = min(min_ny, n.y);
            }
        }
        if (min_ny > reject_t) {
            edge = 0.0;
        }
    }

    if (edge > 0.0 && covered_by_main_pass_only(uv_px)) {
        edge = 0.0;
    }

    let src = textureSample(screen_texture, filtering_sampler, uv_px);
    // A silhouette lies outside its model, on a pixel holding nothing or a
    // farther surface, and is a whole pixel of the model's outline. Its alpha 2
    // (the canvas is float) marks it for the compositor, which sets it at the
    // depth of the nearest surface beside it, so what stands in front of the
    // model hides the model's outline too. Whole, because a line's partial
    // coverage would let the ground show through and the scene grade tints that
    // dark mix (it keeps pure black black); a blended particle drawn over it
    // later leaves its alpha above 1 unless it covers it outright.
    if (outside_line && edge > 0.0) {
        let line_a = edge * resolved_edge_color.a;
        if (src.a <= 0.0) {
            return select(src, vec4f(resolved_edge_color.rgb, 2.0), line_a > 0.0);
        }
        return vec4f(mix(src.rgb, resolved_edge_color.rgb, line_a), 2.0);
    }
    // An empty texel of a transparent canvas takes no line: the compositor lays
    // what has no depth over the scene by its alpha (blended weather, #1139).
    if (src.a <= 0.0) {
        return src;
    }
    // Blend with resolved edge color, respecting its alpha for opacity control.
    var color = mix(src.rgb, resolved_edge_color.rgb, edge * resolved_edge_color.a);

    // Preserve source alpha for compositing (render-to-texture transparency).
    // Where an edge is drawn, force opaque so outlines at entity boundaries are visible.
    return vec4f(color, max(src.a, edge));
}
