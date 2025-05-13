// shader.wgsl

struct VertexOut {
  @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32>,
};

// Bind group 0, binding 0: a single float uniform
@group(0) @binding(0) var<uniform> aspect : f32;          // (width/height)
@group(1) @binding(0) var<uniform> light_pos : vec2<f32>; // (x,y)

@vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOut {
  // Full-screen triangle via vertex_index hack
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0),
                                vec2<f32>(-1.0, 3.0), );
  let p = pos[vi];
  var out : VertexOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  // Map from [-1,1] to [0,1] for uv
  out.uv = p * 0.5 + vec2<f32>(0.5);
  return out;
}

@fragment fn fs_main(in : VertexOut) -> @location(0) vec4<f32> {
  // Construct ray origin / dir in view-space
  let uv = in.uv * 2.0 - vec2<f32>(1.0);
  let fov = 45.0 * 3.14159265 / 180.0;
  let dir = normalize(vec3<f32>(uv.x * aspect, uv.y, -1.0));
  let origin = vec3<f32>(0.0, 0.0, 2.0);

  // Sphere at (0,0,0), radius 1
  let oc = origin;
  let b = dot(oc, dir);
  let c = dot(oc, oc) - 1.0;
  let h = b * b - c;
  if (h < 0.0) {
    // miss → background
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let t = -b - sqrt(h);
  let p = origin + t * dir;
  let n = normalize(p);

  // Simple lambert shading
  let light = normalize(vec3<f32>(light_pos, 1.0));
  let diff = max(dot(n, light), 0.0);
  return vec4<f32>(diff, diff, diff, 1.0);
}

// You’ll need to pass `aspect` as a uniform (width/height).
// Or bake it into the shader via a @group(0) @binding(0) uniform.
