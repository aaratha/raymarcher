// shader.wgsl

const PI : f32 = 3.14159265;
const PHI : f32 = 1.6180339887;

struct VertexOut {
  @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32>,
};

// Bind group 0, binding 0: a single float uniform
@group(0) @binding(0) var<uniform> aspect : f32;            // (width/height)
@group(1) @binding(0) var<uniform> light_pos : vec2<f32>;   // (x,y)
@group(2) @binding(0) var<uniform> orientation : vec4<f32>; // (x,y,z,w)

fn deg_to_rad(deg : f32) -> f32 { return deg * PI / 180.0; }

fn rotate_by_quat(v : vec3<f32>) -> vec3<f32> {
  let qv = orientation.xyz;
  let t = 2.0 * cross(qv, v);
  return v + orientation.w * t + cross(qv, t);
}
fn sdf_sphere(p : vec3<f32>) -> f32 { return length(p) - 1.0; }

fn sdf_box(p : vec3<f32>) -> f32 {
  let b_size = 0.5;
  let b = vec3<f32>(b_size, b_size, b_size);

  // Rotate around arbitrary axis
  let axis = normalize(vec3<f32>(1.0, 1.0, 0.0)); // for example
  let angle = deg_to_rad(45.0);                   // degrees to radians
  let p_rot = p;                                  // rotate_by_quat(p);

  let q = abs(p_rot) - b;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_hollow(p : vec3<f32>) -> f32 {
  // sampling independent computations (only depend on shape)
  let h = 0.2;
  let r = 0.9;
  let t = 0.3;

  let w = sqrt(r * r - h * h);

  // sampling dependant computations
  let q = vec2<f32>(length(p.xz), p.y);
  if (h * q.x < w * q.y) {
    return length(q - vec2(w, h));
  } else {
    return abs(length(q) - r) - t;
  }
}

fn sdf_mandelbulb(p : vec3<f32>) -> f32 {
  var z = p;
  var dr : f32 = 1.0;
  var r : f32 = length(z);
  let power : f32 = 8.0;
  for (var i : u32 = 0u; i < 10u; i = i + 1u) {
    if (r > 2.0) {
      break;
    }
    let theta = acos(z.z / r);
    let phi = atan2(z.y, z.x);
    dr = pow(r, power - 1.0) * power * dr + 1.0;
    let zr = pow(r, power);
    let sinT = sin(power * theta);
    let cosT = cos(power * theta);
    let sinP = sin(power * phi);
    let cosP = cos(power * phi);
    z = vec3<f32>(zr * sinT * cosP, zr * sinT * sinP, zr * cosT) + p;
    r = length(z);
  }
  return 0.5 * log(r) * r / dr;
}

fn sdf_blob(p_in : vec3<f32>) -> f32 {
  // Work on a mutable copy
  var p = abs(p_in);

  // Two “swap‐max” steps
  if (p.x < max(p.y, p.z)) {
    p = p.yzx;
  }
  if (p.x < max(p.y, p.z)) {
    p = p.yzx;
  }

  // Compute the blend factor b
  let b = max(max(dot(p, normalize(vec3<f32>(1.0, 1.0, 1.0))),
                  dot(p.xz, normalize(vec2<f32>(PHI + 1.0, 1.0)))),
              max(dot(p.yx, normalize(vec2<f32>(1.0, PHI))),
                  dot(p.xz, normalize(vec2<f32>(1.0, PHI)))));

  let l = length(p);
  let r = 1.0; // radius
  let a = 0.2; // amplitude
  // Final SDF: base sphere minus a cosine‐modulated “blob” term
  return l - r - a * (r / 2.0) * cos(min(sqrt(1.01 - b / l) * (PI / 0.25), PI));
}

fn sdf(p : vec3<f32>) -> f32 {
  let op = rotate_by_quat(p); // orient_p(p, orientation);

  // return sdf_sphere(op);
  // return sdf_box(op);
  // return sdf_hollow(op);
  // return sdf_mandelbulb(op);
  return sdf_blob(op);
}

fn ray_march(origin : vec3<f32>, dir : vec3<f32>) -> f32 {
  var t = 0.0;
  let max_dist = 100.0;
  let min_dist = 0.001;
  for (var i = 0; i < 100; i++) {
    let p = origin + t * dir;
    let d = sdf(p);
    if (d < min_dist) {
      return t; // hit
    }
    if (t > max_dist) {
      break; // too far
    }
    t = t + d;
  }
  return -1.0; // no hit
}

fn estimate_normal(p : vec3<f32>) -> vec3<f32> {
  let e = 0.001;
  return normalize(vec3<f32>(
      sdf(p + vec3<f32>(e, 0.0, 0.0)) - sdf(p - vec3<f32>(e, 0.0, 0.0)),
      sdf(p + vec3<f32>(0.0, e, 0.0)) - sdf(p - vec3<f32>(0.0, e, 0.0)),
      sdf(p + vec3<f32>(0.0, 0.0, e)) - sdf(p - vec3<f32>(0.0, 0.0, e))));
}

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
  let uv = in.uv * 2.0 - vec2<f32>(1.0);
  let fov = 45.0 * 3.14159265 / 180.0;
  let dir = normalize(vec3<f32>(uv.x * aspect, uv.y, -1.0));
  let origin = vec3<f32>(0.0, 0.0, 2.0);

  let t = ray_march(origin, dir);
  if (t < 0.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // background
  }

  let p = origin + t * dir;
  let n = estimate_normal(p);
  let light = normalize(vec3<f32>(light_pos, 1.0));
  let diff = max(dot(n, light), 0.0);

  return vec4<f32>(diff, diff, diff, 1.0);
}
