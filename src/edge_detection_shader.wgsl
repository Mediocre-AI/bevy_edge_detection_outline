//! Edge Detection Shader
//!
//! This shader implements edge detection based on depth, normal, and color gradients.
//! Three operators are supported via shader defs:
//!   - OPERATOR_SOBEL:        3x3 Sobel filter — 8 samples per type, wider edges, stronger gradients.
//!   - OPERATOR_ROBERTS_CROSS: 2x2 Roberts Cross — 4 samples per type, clean 1px edges.
//!   - OPERATOR_PIXEL_ART:    UDLR 4-direction pairwise comparison — 1px guaranteed, silhouette/crease
//!                            priority, per-entity channel mask (alpha encoding).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View
#import bevy_pbr::view_transformations::uv_to_ndc

@group(0) @binding(0) var screen_texture: texture_2d<f32>;

#ifdef MULTISAMPLED
@group(0) @binding(1) var depth_prepass_texture: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth_prepass_texture: texture_depth_2d;
#endif

#ifdef MULTISAMPLED
@group(0) @binding(2) var normal_prepass_texture: texture_multisampled_2d<f32>;
#else
@group(0) @binding(2) var normal_prepass_texture: texture_2d<f32>;
#endif

@group(0) @binding(3) var filtering_sampler: sampler;
@group(0) @binding(4) var depth_sampler: sampler;

@group(0) @binding(5) var noise_texture: texture_2d<f32>;
@group(0) @binding(6) var noise_sampler: sampler;

@group(0) @binding(7) var<uniform> view: View;
@group(0) @binding(8) var<uniform> ed_uniform: EdgeDetectionUniform;

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

    block_pixel: u32,
    flat_rejection_threshold: f32,
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
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near() / ndc_depth;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return -(view.clip_from_view[3][2] - ndc_depth) / view.clip_from_view[2][2];
#else
    let view_pos = view.view_from_clip * vec4f(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
#endif
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = view.world_from_clip * vec4f(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

fn calculate_view(world_position: vec3f) -> vec3f {
#ifdef VIEW_PROJECTION_ORTHOGRAPHIC
        // Orthographic view vector
        return normalize(vec3f(view.clip_from_world[0].z, view.clip_from_world[1].z, view.clip_from_world[2].z));
#else
        // Only valid for a perspective projection
        return normalize(view.world_position.xyz - world_position.xyz);
#endif
}

// -----------------------
// Depth Detection -------
// -----------------------

fn prepass_depth(uv: vec2f) -> f32 {
#ifdef MULTISAMPLED
    let pixel_coord = vec2i(uv * texture_size);
    let depth = textureLoad(depth_prepass_texture, pixel_coord, sample_index_i);
#else
    let depth = textureSample(depth_prepass_texture, depth_sampler, uv);
#endif
    return depth;
}

fn prepass_view_z(uv: vec2f) -> f32 {
    let depth = prepass_depth(uv);
    return depth_ndc_to_view_z(depth);
}

fn detect_edge_depth(uv: vec2f, thickness: f32, fresnel: f32) -> f32 {
    let offset = texel_size * thickness;

#ifdef OPERATOR_SOBEL
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
#else
    // 2x2 Roberts Cross: diagonal differences from 4 samples
    let d00 = prepass_view_z(uv);
    let d10 = prepass_view_z(uv + vec2f(offset.x, 0.0));
    let d01 = prepass_view_z(uv + vec2f(0.0, offset.y));
    let d11 = prepass_view_z(uv + offset);

    let diff_diag0 = d00 - d11;
    let diff_diag1 = d10 - d01;
    let grad = max(abs(diff_diag0), abs(diff_diag1));
    let view_z = abs(d00);
#endif

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
    return prepass_normal_raw(uv).xyz;
}

fn prepass_normal_raw(uv: vec2f) -> vec4f {
#ifdef MULTISAMPLED
    let pixel_coord = vec2i(uv * texture_size);
    let normal = textureLoad(normal_prepass_texture, pixel_coord, sample_index_i);
#else
    let normal = textureSample(normal_prepass_texture, filtering_sampler, uv);
#endif
    return normal;
}

fn detect_edge_normal(uv: vec2f, thickness: f32) -> f32 {
    let offset = texel_size * thickness;

#ifdef OPERATOR_SOBEL
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
#else
    let n00 = prepass_normal(uv);
    let n10 = prepass_normal(uv + vec2f(offset.x, 0.0));
    let n01 = prepass_normal(uv + vec2f(0.0, offset.y));
    let n11 = prepass_normal(uv + offset);

    let diff0 = n00 - n11;
    let diff1 = n10 - n01;
    let grad = sqrt(dot(diff0, diff0) + dot(diff1, diff1));
#endif

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

#ifdef OPERATOR_SOBEL
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
#else
    let c00 = prepass_color(uv);
    let c10 = prepass_color(uv + vec2f(offset.x, 0.0));
    let c01 = prepass_color(uv + vec2f(0.0, offset.y));
    let c11 = prepass_color(uv + offset);

    let diff0 = c00 - c11;
    let diff1 = c10 - c01;
    let grad = sqrt(dot(diff0, diff0) + dot(diff1, diff1));
#endif

    return f32(grad > ed_uniform.color_threshold);
}

// -----------------------
// Channel Mask Decoding --
// -----------------------

/// Decode the normal prepass alpha into (enable_silhouette, enable_crease).
/// Alpha encoding: 0.0=SKIP, 0.25=SILHOUETTE_ONLY, 0.50=CREASE_ONLY, 0.75~1.0=BOTH
fn decode_edge_mask(alpha: f32) -> vec2<bool> {
    if (alpha < 0.125) { return vec2<bool>(false, false); }
    else if (alpha < 0.375) { return vec2<bool>(true, false); }
    else if (alpha < 0.625) { return vec2<bool>(false, true); }
    else { return vec2<bool>(true, true); }
}

// -----------------------
// PixelArt Operator ------
// -----------------------

#ifdef OPERATOR_PIXEL_ART

/// UDLR pairwise normal comparison for crease detection.
/// Directional tie-breaking: only marks one side to ensure 1px crease edges.
/// If the neighbor is SKIP (alpha < 0.125), center always wins — otherwise
/// the edge would be lost because the SKIP side never draws edges.
fn check_crease(uv: vec2f, offset: vec2f) -> bool {
    let n_center = prepass_normal(uv);
    let raw_neighbor = prepass_normal_raw(uv + offset);
    let n_neighbor = raw_neighbor.xyz;
    let diff = n_center - n_neighbor;
    if (dot(diff, diff) <= ed_uniform.normal_threshold * ed_uniform.normal_threshold) {
        return false;
    }
    // If neighbor is SKIP, center always wins (SKIP side won't draw any edge).
    if (raw_neighbor.a < 0.125) {
        return true;
    }
    // Tie-break: only mark on the pixel with larger component sum.
    let sum_c = n_center.x + n_center.y + n_center.z;
    let sum_n = n_neighbor.x + n_neighbor.y + n_neighbor.z;
    return sum_c >= sum_n;
}

#endif

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
#ifdef MULTISAMPLED
    @builtin(sample_index) sample_index: u32,
#endif
    in: FullscreenVertexOutput
) -> @location(0) vec4f {
#ifdef MULTISAMPLED
    sample_index_i = i32(sample_index);
#endif

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
    var resolved_edge_color = ed_uniform.edge_color;

#ifdef OPERATOR_PIXEL_ART
    // PixelArt operator: UDLR pairwise comparison with silhouette/crease priority.
    let px_offset = vec2f(block_pixel, block_pixel) / texture_size;

    let offsets = array<vec2f, 4>(
        vec2f(0.0, px_offset.y),   // Down (+y in UV space)
        vec2f(0.0, -px_offset.y),  // Up
        vec2f(-px_offset.x, 0.0),  // Left
        vec2f(px_offset.x, 0.0),   // Right
    );

    // Decode per-entity edge mask from normal prepass alpha
    let mask = decode_edge_mask(prepass_normal_raw(uv_noise_px).a);

    var is_silhouette = false;
    var is_crease = false;
    var at_depth_boundary = false;

    // 1) Silhouette search — directional: only foreground pixel gets the edge.
    //    Also tracks if we're at ANY depth boundary (either side) to suppress crease.
#ifdef ENABLE_DEPTH
    if (mask.x) {
        let center_z = prepass_view_z(uv_noise_px);
        let view_z = abs(center_z);
        let steep_adj = smoothstep(ed_uniform.steep_angle_threshold, 1.0, fresnel)
                        * ed_uniform.steep_angle_multiplier * view_z;
        let threshold = ed_uniform.depth_threshold * (1.0 + steep_adj);

        for (var i = 0; i < 4; i++) {
            let neighbor_z = prepass_view_z(uv_noise_px + offsets[i]);
            let diff = center_z - neighbor_z;
            if (diff > threshold) {
                // Center is closer → foreground silhouette
                is_silhouette = true;
                at_depth_boundary = true;
                break;
            }
            if (-diff > threshold) {
                // Center is farther → background side of a boundary.
                // Only suppress crease if the foreground neighbor is non-SKIP,
                // because a SKIP neighbor won't draw its own silhouette edge.
                let neighbor_alpha = prepass_normal_raw(uv_noise_px + offsets[i]).a;
                if (neighbor_alpha >= 0.125) {
                    at_depth_boundary = true;
                }
            }
        }
    }
#endif

    // 2) Crease search — skip if silhouette found OR at a depth boundary.
    //    Suppressing crease at depth boundaries prevents the background pixel
    //    from drawing a false crease (different objects = different normals).
#ifdef ENABLE_NORMAL
    if (!is_silhouette && !at_depth_boundary && mask.y) {
        for (var j = 0; j < 4; j++) {
            if (check_crease(uv_noise_px, offsets[j])) {
                is_crease = true;
                break;
            }
        }
    }
#endif

    // 3) Type-specific color
    if (is_silhouette) {
        edge = 1.0;
        resolved_edge_color = ed_uniform.silhouette_color;
    } else if (is_crease) {
        edge = 1.0;
        resolved_edge_color = ed_uniform.crease_color;
    }

#ifdef ENABLE_COLOR
    if (edge < 1.0) {
        let edge_color_val = detect_edge_color(uv_noise_px, ed_uniform.color_thickness);
        if (edge_color_val > 0.0) {
            edge = 1.0;
            resolved_edge_color = ed_uniform.edge_color;
        }
    }
#endif

#else
    // Sobel / Roberts Cross operators with silhouette/crease priority.
#ifdef ENABLE_DEPTH
    let edge_depth = detect_edge_depth(uv_noise_px, ed_uniform.depth_thickness, fresnel);
    if (edge_depth > 0.0) {
        edge = 1.0;
        resolved_edge_color = ed_uniform.silhouette_color;
    }
#endif

#ifdef ENABLE_NORMAL
    if (edge < 1.0) {
        let edge_normal = detect_edge_normal(uv_noise_px, ed_uniform.normal_thickness);
        if (edge_normal > 0.0) {
            edge = 1.0;
            resolved_edge_color = ed_uniform.crease_color;
        }
    }
#endif

#ifdef ENABLE_COLOR
    if (edge < 1.0) {
        let edge_color_val = detect_edge_color(uv_noise_px, ed_uniform.color_thickness);
        if (edge_color_val > 0.0) {
            edge = 1.0;
            resolved_edge_color = ed_uniform.edge_color;
        }
    }
#endif
#endif  // OPERATOR_PIXEL_ART

    // Edge mask: suppress edges on pixels marked with alpha=0.0 in normal prepass.
    // Materials using the NoEdgeExtension write alpha=0.0 (e.g. hex tile surfaces).
    // Standard materials write alpha=1.0 (walls, settlements, flags, armies).
    // Check the center pixel + immediate neighbors: if ALL have mask=0, suppress.
    //
    // NOTE: PixelArt operator handles masking via decode_edge_mask() which supports
    // the 4-level alpha encoding (SKIP/SILHOUETTE_ONLY/CREASE_ONLY/BOTH).
    // This legacy suppression uses alpha < 0.5 as "no-edge", which would incorrectly
    // suppress SILHOUETTE_ONLY (alpha=0.25). Skip it for PixelArt.
#ifndef OPERATOR_PIXEL_ART
    if (edge > 0.0) {
        let center_raw = prepass_normal_raw(uv_noise_px);
        if (center_raw.a < 0.5) {
            // Center pixel is no-edge. Check if all neighbors are also no-edge.
            let max_thickness = max(ed_uniform.depth_thickness, ed_uniform.normal_thickness);
            var max_alpha = center_raw.a;
            for (var iy = -1; iy <= 1; iy++) {
                for (var ix = -1; ix <= 1; ix++) {
                    let offset_uv = uv_noise_px + texel_size * vec2f(f32(ix), f32(iy)) * max_thickness;
                    let raw = prepass_normal_raw(offset_uv);
                    max_alpha = max(max_alpha, raw.a);
                }
            }
            // If all pixels in neighborhood are no-edge (alpha < 0.5), suppress
            if (max_alpha < 0.5) {
                edge = 0.0;
            }
        }
    }
#endif

    // Flat surface rejection (fallback for StandardMaterial entities without custom prepass)
    if (ed_uniform.flat_rejection_threshold > 0.0 && edge > 0.0) {
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

    let src = textureSample(screen_texture, filtering_sampler, uv_px);
    // Blend with resolved edge color, respecting its alpha for opacity control.
    var color = mix(src.rgb, resolved_edge_color.rgb, edge * resolved_edge_color.a);

    // Preserve source alpha for compositing (render-to-texture transparency).
    // Where an edge is drawn, force opaque so outlines at entity boundaries are visible.
    return vec4f(color, max(src.a, edge));
}