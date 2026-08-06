/* Client-only WebGPU screen-space fluid reconstruction for gameplay mode. */

export const SURFACE_FRAME_VERSION = 1;
export const SURFACE_STRIDE_FLOATS = 16;

const renderers = new Map();
let deferredControls = {};

export function decodeSurfaceFrame(frame) {
  if (!frame || frame.version !== SURFACE_FRAME_VERSION) {
    throw new Error("Unsupported or missing surface-frame version");
  }
  if (!Number.isInteger(frame.count) || frame.count < 0) {
    throw new Error("Invalid surface-frame particle count");
  }
  if (frame.stride_floats !== SURFACE_STRIDE_FLOATS) {
    throw new Error("Invalid surface-frame stride");
  }
  const binary = atob(frame.payload || "");
  const expected = frame.count * SURFACE_STRIDE_FLOATS * 4;
  if (binary.length !== expected) {
    throw new Error(`Surface payload has ${binary.length} bytes; expected ${expected}`);
  }
  const bytes = new Uint8Array(expected);
  for (let i = 0; i < expected; i += 1) bytes[i] = binary.charCodeAt(i);
  const floats = new Float32Array(bytes.buffer);
  for (let i = 0; i < floats.length; i += 1) {
    if (!Number.isFinite(floats[i])) throw new Error("Surface payload is non-finite");
  }
  return floats;
}

export function perspectiveZO(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ]);
}

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize3 = (v) => {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
};

export function lookAt(eye, target, up = [0, 0, 1]) {
  const z = normalize3(sub3(eye, target));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ]);
}

export function multiplyMat4(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let k = 0; k < 4; k += 1) value += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = value;
    }
  }
  return out;
}

export function invertMat4(a) {
  const out = new Float32Array(16);
  const m = Array.from(a);
  const inv = [
    m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10],
    -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10],
    m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6],
    -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6],
    -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10],
    m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10],
    -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6],
    m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6],
    m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9],
    -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9],
    m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5],
    -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5],
    -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9],
    m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9],
    -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5],
    m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5],
  ];
  let det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
  if (Math.abs(det) < 1e-12) throw new Error("Singular camera matrix");
  det = 1 / det;
  for (let i = 0; i < 16; i += 1) out[i] = inv[i] * det;
  return out;
}

export function webGPUAvailable(scope = globalThis) {
  return Boolean(scope.navigator && scope.navigator.gpu);
}

export function canvasPixelSize(width, height, devicePixelRatio = 1) {
  const dpr = Math.min(Math.max(Number(devicePixelRatio) || 1, 1), 2);
  return [Math.max(1, Math.floor(width * dpr)), Math.max(1, Math.floor(height * dpr))];
}

export const SURFACE_WGSL = /* wgsl */ `
struct Uniforms {
  view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>,
  camera: vec4<f32>, viewport: vec4<f32>,
  camera_right: vec4<f32>, camera_up: vec4<f32>,
  tank_min: vec4<f32>, tank_max: vec4<f32>,
  body_center: vec4<f32>, body_quat: vec4<f32>, body_half: vec4<f32>,
  water: vec4<f32>, params: vec4<f32>, light: vec4<f32>, misc: vec4<f32>,
};
struct Particle { center_speed: vec4<f32>, axis0: vec4<f32>, axis1: vec4<f32>, axis2: vec4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var thickness_tex: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;

struct SplatOut { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) index: u32, @location(1) @interpolate(flat) speed: f32 };
@vertex fn vs_splat(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SplatOut {
  let corners = array<vec2<f32>, 6>(vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(-1,1),vec2(1,-1),vec2(1,1));
  let p = particles[instance];
  let center_clip = u.view_proj * vec4(p.center_speed.xyz, 1.0);
  let radius = max(length(p.axis0.xyz), max(length(p.axis1.xyz), length(p.axis2.xyz))) * u.params.w;
  let edge_clip = u.view_proj * vec4(p.center_speed.xyz + u.camera_right.xyz * radius, 1.0);
  let pixel_radius = max(1.5, length((edge_clip.xy / edge_clip.w - center_clip.xy / center_clip.w) * u.viewport.xy * 0.5));
  var out: SplatOut;
  out.position = center_clip;
  let offset = corners[vertex] * (pixel_radius * 2.0 / u.viewport.xy) * center_clip.w;
  out.position = vec4(out.position.xy + offset, out.position.zw);
  out.index = instance; out.speed = p.center_speed.w;
  return out;
}

fn world_ray(position: vec2<f32>) -> vec3<f32> {
  let ndc = vec2(position.x / u.viewport.x * 2.0 - 1.0, 1.0 - position.y / u.viewport.y * 2.0);
  var far = u.inv_view_proj * vec4(ndc, 1.0, 1.0); far /= far.w;
  return normalize(far.xyz - u.camera.xyz);
}
fn basis_mul_inv(a: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
  let det = dot(a[0], cross(a[1], a[2]));
  return vec3(dot(v,cross(a[1],a[2])), dot(v,cross(a[2],a[0])), dot(v,cross(a[0],a[1]))) / det;
}
fn ellipsoid_hit(index: u32, position: vec2<f32>) -> vec4<f32> {
  let p = particles[index];
  let ray = world_ray(position);
  let basis = mat3x3(p.axis0.xyz * u.params.w, p.axis1.xyz * u.params.w, p.axis2.xyz * u.params.w);
  let ro = basis_mul_inv(basis, u.camera.xyz - p.center_speed.xyz);
  let rd = basis_mul_inv(basis, ray);
  let b = dot(ro, rd); let c = dot(ro, ro) - 1.0; let a = dot(rd, rd);
  let disc = b*b - a*c;
  if (disc < 0.0) { return vec4(-1.0); }
  let root = sqrt(disc); let t0 = (-b-root)/a; let t1 = (-b+root)/a;
  if (t1 <= 0.0) { return vec4(-1.0); }
  return vec4(max(t0, 0.0), t1, ray.xy);
}
struct DepthOut { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 };
@fragment fn fs_depth(in: SplatOut) -> DepthOut {
  let hit = ellipsoid_hit(in.index, in.position.xy); if (hit.x < 0.0) { discard; }
  let ray = world_ray(in.position.xy); let world = u.camera.xyz + ray * hit.x;
  let clip = u.view_proj * vec4(world, 1.0);
  var out: DepthOut; out.color = vec4(hit.x, in.speed, 0.0, 1.0); out.depth = clip.z / clip.w; return out;
}
@fragment fn fs_thickness(in: SplatOut) -> @location(0) vec4<f32> {
  let hit = ellipsoid_hit(in.index, in.position.xy); if (hit.x < 0.0) { discard; }
  return vec4((hit.y-hit.x) * u.params.z, 0.0, 0.0, 1.0);
}

struct FullOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs_full(@builtin(vertex_index) i: u32) -> FullOut {
  let p = array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  var out: FullOut; out.position=vec4(p[i],0,1); out.uv=vec2((p[i].x+1.0)*0.5,(1.0-p[i].y)*0.5); return out;
}
@fragment fn fs_smooth(in: FullOut) -> @location(0) vec4<f32> {
  let center = textureSampleLevel(depth_tex, linear_sampler, in.uv, 0.0);
  if (center.r <= 0.0) { return vec4(0.0); }
  let direction = u.misc.zw / u.viewport.xy;
  let radius = i32(clamp(u.misc.y, 1.0, 8.0));
  var depth_sum = center.r;
  var speed_sum = center.g;
  var weights = 1.0;
  for (var offset = 1; offset <= 8; offset += 1) {
    if (offset > radius) { break; }
    let spatial = exp(-f32(offset*offset) / max(2.0*f32(radius*radius),1.0));
    for (var side = -1; side <= 1; side += 2) {
      let sample = textureSampleLevel(depth_tex, linear_sampler, in.uv + direction*f32(offset*side), 0.0);
      if (sample.r > 0.0) {
        // Preserve the silhouette by ignoring empty pixels, but deliberately
        // smooth across neighbouring splat fronts. A steep bilateral range
        // term leaves one rounded ridge per particle and reads as capsules.
        let delta = sample.r-center.r;
        let range = exp(-delta*delta * 0.75);
        let w = spatial*range;
        depth_sum += sample.r*w; speed_sum += sample.g*w; weights += w;
      }
    }
  }
  return vec4(depth_sum/weights, speed_sum/weights, 0.0, 1.0);
}
fn ray_for_uv(uv: vec2<f32>) -> vec3<f32> { return world_ray(uv * u.viewport.xy); }
fn sky(ray: vec3<f32>) -> vec3<f32> {
  let horizon = pow(clamp(1.0-abs(ray.z),0.0,1.0),5.0);
  let altitude = pow(clamp(ray.z,0.0,1.0),0.65);
  let sun = pow(max(dot(ray, normalize(u.light.xyz)),0.0), 420.0);
  return mix(vec3(0.42,0.60,0.71),vec3(0.025,0.07,0.15),altitude)
      + horizon*vec3(0.12,0.14,0.15) + sun*vec3(7.0,5.6,3.5);
}
fn quat_rotate(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn body_hit(ray: vec3<f32>) -> vec4<f32> {
  if (u.body_center.w < 0.5) { return vec4(0.0,0.0,0.0,-1.0); }
  let qi=vec4(-u.body_quat.xyz,u.body_quat.w);
  let ro=quat_rotate(u.camera.xyz-u.body_center.xyz,qi);
  let rd=quat_rotate(ray,qi);
  let inverse=1.0/(rd+sign(rd)*vec3(1.0e-7));
  let ta=(-u.body_half.xyz-ro)*inverse; let tb=(u.body_half.xyz-ro)*inverse;
  let near3=min(ta,tb); let far3=max(ta,tb);
  let near=max(max(near3.x,near3.y),near3.z); let far=min(min(far3.x,far3.y),far3.z);
  if (far<max(near,0.0)) { return vec4(0.0,0.0,0.0,-1.0); }
  let t=max(near,0.0); let local=ro+rd*t; let face=abs(local/u.body_half.xyz);
  var normal=vec3(0.0);
  if(face.x>face.y&&face.x>face.z){normal=vec3(sign(local.x),0,0);}else if(face.y>face.z){normal=vec3(0,sign(local.y),0);}else{normal=vec3(0,0,sign(local.z));}
  return vec4(normalize(quat_rotate(normal,u.body_quat)),t);
}
fn shade_body(hit: vec4<f32>, ray: vec3<f32>) -> vec3<f32> {
  let diffuse=0.25+0.75*max(dot(hit.xyz,normalize(u.light.xyz)),0.0);
  let rim=pow(1.0-max(dot(hit.xyz,-ray),0.0),3.0);
  return vec3(0.96,0.28,0.055)*diffuse+vec3(1.0,0.42,0.12)*rim*0.35;
}
fn background(uv: vec2<f32>, ray: vec3<f32>) -> vec3<f32> {
  var color = sky(ray);
  if (ray.z < -0.0001) {
    let t = -u.camera.z/ray.z; let p = u.camera.xyz + ray*t;
    let checker=select(0.0,1.0,(i32(floor(p.x*2.0))+i32(floor(p.y*2.0)))%2==0);
    color=mix(vec3(0.025,0.038,0.052),vec3(0.045,0.068,0.085),checker);
    if (p.x>=u.tank_min.x && p.x<=u.tank_max.x && p.y>=u.tank_min.y && p.y<=u.tank_max.y) {
      let gx=abs(fract(p.x*5.0)-0.5); let gy=abs(fract((p.y-u.tank_min.y)*10.0)-0.5);
      let grid=1.0-smoothstep(0.44,0.49,max(gx,gy));
      color=mix(vec3(0.025,0.055,0.085),vec3(0.08,0.25,0.31),grid*0.42);
    }
  }
  let body=body_hit(ray); if(body.w>0.0){color=shade_body(body,ray);}
  return color;
}
fn position_at(uv: vec2<f32>, depth: f32) -> vec3<f32> { return u.camera.xyz + ray_for_uv(uv)*depth; }
fn filtered_thickness(uv: vec2<f32>) -> f32 {
  let px=vec2(6.0/u.viewport.x,0.0); let py=vec2(0.0,6.0/u.viewport.y);
  let px2=px*2.0; let py2=py*2.0;
  var sum=textureSampleLevel(thickness_tex,linear_sampler,uv,0.0).r*4.0;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv+px,0.0).r*2.0;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv-px,0.0).r*2.0;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv+py,0.0).r*2.0;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv-py,0.0).r*2.0;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv+px+py,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv+px-py,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv-px+py,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv-px-py,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv+px2,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv-px2,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv+py2,0.0).r;
  sum+=textureSampleLevel(thickness_tex,linear_sampler,uv-py2,0.0).r;
  return sum/20.0;
}
@fragment fn fs_composite(in: FullOut) -> @location(0) vec4<f32> {
  let sample = textureSampleLevel(depth_tex,linear_sampler,in.uv,0.0);
  let ray = ray_for_uv(in.uv);
  let thickness=filtered_thickness(in.uv);
  let edge=smoothstep(0.025,0.12,thickness);
  let body=body_hit(ray);
  if(body.w>0.0&&(edge<=0.05||body.w<sample.r)){return vec4(pow(shade_body(body,ray),vec3(1.0/2.2)),1.0);}
  let backdrop=background(in.uv,ray);
  if (sample.r<=0.0||edge<=0.01) { return vec4(backdrop,1.0); }
  // A wider derivative footprint suppresses rgba16float depth quantisation
  // without softening the already reconstructed silhouette.
  let px=vec2(6.0/u.viewport.x,0.0); let py=vec2(0.0,6.0/u.viewport.y);
  let dl=textureSampleLevel(depth_tex,linear_sampler,in.uv-px,0.0).r;
  let dr=textureSampleLevel(depth_tex,linear_sampler,in.uv+px,0.0).r;
  let du=textureSampleLevel(depth_tex,linear_sampler,in.uv-py,0.0).r;
  let dd=textureSampleLevel(depth_tex,linear_sampler,in.uv+py,0.0).r;
  let pl=position_at(in.uv-px,select(sample.r,dl,dl>0.0)); let pr=position_at(in.uv+px,select(sample.r,dr,dr>0.0));
  let pu=position_at(in.uv-py,select(sample.r,du,du>0.0)); let pd=position_at(in.uv+py,select(sample.r,dd,dd>0.0));
  var normal=normalize(cross(pr-pl,pd-pu)); if(dot(normal,-ray)<0.0){normal=-normal;}
  let world=position_at(in.uv,sample.r);
  let phase=u.tank_min.w;
  let ripple=vec3(
      sin(world.x*18.0+world.y*11.0+phase*2.1)*0.018,
      cos(world.x*13.0-world.y*21.0-phase*1.7)*0.016,
      0.0);
  normal=normalize(normal+ripple);
  if (u.misc.x==1.0) { return vec4(vec3(sample.r/max(u.viewport.w,0.001)),1.0); }
  if (u.misc.x==2.0) { return vec4(vec3(1.0-exp(-thickness*3.0)),1.0); }
  if (u.misc.x==3.0) { return vec4(normal*0.5+0.5,1.0); }
  let view=-ray; let ndv=max(dot(normal,view),0.0);
  let fresnel=0.025+0.42*pow(1.0-ndv,5.0);
  let refract_uv=clamp(in.uv+normal.xy*u.params.x*min(thickness,1.5),vec2(0.002),vec2(0.998));
  let refracted=background(refract_uv,ray_for_uv(refract_uv));
  let sigma=(vec3(1.08)-u.water.rgb*0.85)*u.water.a;
  let absorption=exp(-sigma*max(thickness,0.0));
  let transmitted=refracted*absorption + vec3(0.008,0.08,0.13)*(1.0-absorption);
  let reflected=sky(reflect(ray,normal))*0.55;
  let halfv=normalize(normalize(u.light.xyz)+view);
  let spec=pow(max(dot(normal,halfv),0.0),mix(75.0,16.0,u.params.y))*0.12;
  let foam=smoothstep(3.5,7.0,sample.g)*smoothstep(0.62,0.90,1.0-ndv)*0.12;
  let scatter=u.water.rgb*0.055*(1.0-exp(-thickness*1.4));
  let color=mix(transmitted,reflected,clamp(fresnel+u.params.y*0.03,0.0,0.48))+spec+foam+scatter;
  return vec4(pow(mix(backdrop,color,edge),vec3(1.0/2.2)),1.0);
}`;

const pushTiming = (array, value) => { array.push(value); if (array.length > 180) array.shift(); };
const percentile = (values, p) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
};

function parseColor(hex) {
  const value = /^#[0-9a-f]{6}$/i.test(hex || "") ? hex.slice(1) : "169dc4";
  return [0, 2, 4].map((index) => parseInt(value.slice(index, index + 2), 16) / 255);
}

async function findCanvas() {
  for (let i = 0; i < 60; i += 1) {
    const canvas = document.getElementById("gameplay-surface-canvas");
    if (canvas) return canvas;
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  throw new Error("WebGPU surface canvas was not mounted");
}

class SurfaceRenderer {
  constructor(canvas) {
    this.canvas = canvas; this.controls = {...deferredControls}; this.frame = null;
    this.camera = {target:[2.45,0,0.38], yaw:-1.36, pitch:0.34, distance:5.9};
    this.uploadTimes=[]; this.renderTimes=[]; this.dropped=0; this.resetRequested=false;
    this.pointer=null; this.needsRender=false;
  }
  async initialize() {
    if (!webGPUAvailable()) throw new Error("This browser does not expose WebGPU");
    this.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
    if (!this.adapter) throw new Error("No WebGPU adapter is available");
    this.device=await this.adapter.requestDevice();
    this.device.lost.then((info)=>{ this.failed=`WebGPU device lost: ${info.message || info.reason}`; });
    this.context=this.canvas.getContext("webgpu");
    this.format=navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"});
    this.module=this.device.createShaderModule({label:"SPH surface WGSL",code:SURFACE_WGSL});
    const compilation=await this.module.getCompilationInfo();
    const shaderErrors=compilation.messages.filter((message)=>message.type==="error");
    if(shaderErrors.length){throw new Error(shaderErrors.map((message)=>`WGSL ${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));}
    const uniformSize=84*4;
    this.uniformBuffer=this.device.createBuffer({size:uniformSize,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.uniformBufferH=this.device.createBuffer({size:uniformSize,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.uniformBufferV=this.device.createBuffer({size:uniformSize,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.particleCapacity=1;
    this.particleBuffer=this.device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.sampler=this.device.createSampler({magFilter:"linear",minFilter:"linear"});
    this.depthPipeline=await this.device.createRenderPipelineAsync({layout:"auto",vertex:{module:this.module,entryPoint:"vs_splat"},fragment:{module:this.module,entryPoint:"fs_depth",targets:[{format:"rgba16float"}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
    this.thicknessPipeline=await this.device.createRenderPipelineAsync({layout:"auto",vertex:{module:this.module,entryPoint:"vs_splat"},fragment:{module:this.module,entryPoint:"fs_thickness",targets:[{format:"rgba16float",blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}});
    this.smoothPipeline=await this.device.createRenderPipelineAsync({layout:"auto",vertex:{module:this.module,entryPoint:"vs_full"},fragment:{module:this.module,entryPoint:"fs_smooth",targets:[{format:"rgba16float"}]},primitive:{topology:"triangle-list"}});
    this.compositePipeline=await this.device.createRenderPipelineAsync({layout:"auto",vertex:{module:this.module,entryPoint:"vs_full"},fragment:{module:this.module,entryPoint:"fs_composite",targets:[{format:this.format}]},primitive:{topology:"triangle-list"}});
    this.installCamera();
    this.resizeObserver=new ResizeObserver(()=>this.scheduleRender()); this.resizeObserver.observe(this.canvas);
  }
  installCamera() {
    this.canvas.addEventListener("contextmenu",(event)=>event.preventDefault());
    this.canvas.addEventListener("pointerdown",(event)=>{this.pointer={x:event.clientX,y:event.clientY,button:event.button};try{this.canvas.setPointerCapture(event.pointerId);}catch(_error){/* Synthetic tests may not own an active pointer. */}});
    this.canvas.addEventListener("pointerup",()=>{this.pointer=null;});
    this.canvas.addEventListener("pointermove",(event)=>{
      if(!this.pointer)return; const dx=event.clientX-this.pointer.x,dy=event.clientY-this.pointer.y; this.pointer.x=event.clientX;this.pointer.y=event.clientY;
      if(this.pointer.button===0&&!event.shiftKey){this.camera.yaw-=dx*0.006;this.camera.pitch=Math.max(-0.15,Math.min(1.25,this.camera.pitch+dy*0.005));}
      else {const scale=this.camera.distance*0.0018;this.camera.target[0]-=dx*scale;this.camera.target[2]+=dy*scale;}
      this.scheduleRender();
    });
    this.canvas.addEventListener("wheel",(event)=>{event.preventDefault();this.camera.distance=Math.max(1.4,Math.min(14,this.camera.distance*Math.exp(event.deltaY*0.001)));this.scheduleRender();},{passive:false});
    this.canvas.addEventListener("dblclick",()=>this.resetCamera());
  }
  resetCamera(){this.camera={target:[2.45,0,0.38],yaw:-1.36,pitch:0.34,distance:5.9};this.scheduleRender();}
  scheduleRender(){if(this.needsRender)return;this.needsRender=true;requestAnimationFrame(()=>{this.needsRender=false;if(this.frame)this.render();});}
  ensureSize() {
    const [width,height]=canvasPixelSize(this.canvas.clientWidth,this.canvas.clientHeight,window.devicePixelRatio);
    if(this.canvas.width===width&&this.canvas.height===height&&this.depthTexture)return;
    this.canvas.width=width;this.canvas.height=height;
    for(const name of ["depthTexture","smoothA","smoothB","thicknessTexture","depthAttachment"]){if(this[name])this[name].destroy();}
    const texture=(format,usage)=>this.device.createTexture({size:[width,height],format,usage});
    const usage=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING;
    this.depthTexture=texture("rgba16float",usage);this.smoothA=texture("rgba16float",usage);this.smoothB=texture("rgba16float",usage);this.thicknessTexture=texture("rgba16float",usage);this.depthAttachment=texture("depth24plus",GPUTextureUsage.RENDER_ATTACHMENT);
  }
  ensureParticles(byteLength){if(byteLength<=this.particleCapacity)return;this.particleBuffer.destroy();this.particleCapacity=Math.max(64,2**Math.ceil(Math.log2(byteLength)));this.particleBuffer=this.device.createBuffer({size:this.particleCapacity,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});}
  updateUniforms(direction=[0,0], target=this.uniformBuffer) {
    const width=this.canvas.width,height=this.canvas.height; const c=this.camera;
    const cp=Math.cos(c.pitch); const eye=[c.target[0]+c.distance*cp*Math.cos(c.yaw),c.target[1]+c.distance*cp*Math.sin(c.yaw),c.target[2]+c.distance*Math.sin(c.pitch)];
    const view=lookAt(eye,c.target), projection=perspectiveZO(Math.PI/4,width/height,0.02,30), vp=multiplyMat4(projection,view),inv=invertMat4(vp);
    const right=[view[0],view[4],view[8]],up=[view[1],view[5],view[9]]; const values=new Float32Array(84);values.set(vp,0);values.set(inv,16);
    values.set([...eye,1],32);values.set([width,height,0.02,30],36);values.set([...right,0],40);values.set([...up,0],44);
    const bounds=this.frame.tank_bounds||[0,5.3667,-.25,.25,0,1.5];values.set([bounds[0],bounds[2],bounds[4],Number(this.frame.time||0)],48);values.set([bounds[1],bounds[3],bounds[5],0],52);
    const body=this.frame.body||{};values.set([...(body.center||[0,0,0]),body.visible&&this.controls.obstacle_visible!==false?1:0],56);values.set(body.orientation||[0,0,0,1],60);values.set([...(body.half_size||[.16,.14,.1]),0],64);
    const color=parseColor(this.controls.water_color);values.set([...color,Number(this.controls.absorption??1.8)],68);values.set([Number(this.controls.refraction??.035),Number(this.controls.roughness??.1),Number(this.controls.thickness??.85),Number(this.controls.splat_scale??1)],72);
    values.set([-.35,-.55,.76,0],76);
    values.set([({final:0,depth:1,thickness:2,normals:3})[this.controls.debug_view]??0,Number(this.controls.smoothing_radius??4),direction[0],direction[1]],80);
    this.device.queue.writeBuffer(target,0,values);
  }
  bind(pipeline, depth, thickness, uniform=this.uniformBuffer) {return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},...(pipeline===this.depthPipeline||pipeline===this.thicknessPipeline?[{binding:1,resource:{buffer:this.particleBuffer}}]:[]),...(pipeline===this.smoothPipeline||pipeline===this.compositePipeline?[{binding:2,resource:depth.createView()}]:[]),...(pipeline===this.compositePipeline?[{binding:3,resource:thickness.createView()}]:[]),...(pipeline===this.smoothPipeline||pipeline===this.compositePipeline?[{binding:4,resource:this.sampler}]:[])]});}
  async setFrame(frame,floats) {this.frame=frame;this.ensureSize();this.ensureParticles(floats.byteLength);const start=performance.now();this.device.queue.writeBuffer(this.particleBuffer,0,floats);pushTiming(this.uploadTimes,performance.now()-start);await this.render();}
  async render() {
    if(!this.frame||this.failed)return;this.ensureSize();this.updateUniforms();const start=performance.now();const encoder=this.device.createCommandEncoder({label:"SPH fluid surface"});
    let pass=encoder.beginRenderPass({colorAttachments:[{view:this.depthTexture.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthAttachment.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});pass.setPipeline(this.depthPipeline);pass.setBindGroup(0,this.bind(this.depthPipeline,this.depthTexture,this.thicknessTexture));pass.draw(6,this.frame.count);pass.end();
    pass=encoder.beginRenderPass({colorAttachments:[{view:this.thicknessTexture.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});pass.setPipeline(this.thicknessPipeline);pass.setBindGroup(0,this.bind(this.thicknessPipeline,this.depthTexture,this.thicknessTexture));pass.draw(6,this.frame.count);pass.end();
    let current=this.depthTexture;const iterations=Math.max(0,Math.min(4,Number(this.controls.smoothing_iterations??2)));
    for(let i=0;i<iterations;i+=1){this.updateUniforms([1,0],this.uniformBufferH);pass=encoder.beginRenderPass({colorAttachments:[{view:this.smoothA.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});pass.setPipeline(this.smoothPipeline);pass.setBindGroup(0,this.bind(this.smoothPipeline,current,this.thicknessTexture,this.uniformBufferH));pass.draw(3);pass.end();this.updateUniforms([0,1],this.uniformBufferV);pass=encoder.beginRenderPass({colorAttachments:[{view:this.smoothB.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});pass.setPipeline(this.smoothPipeline);pass.setBindGroup(0,this.bind(this.smoothPipeline,this.smoothA,this.thicknessTexture,this.uniformBufferV));pass.draw(3);pass.end();current=this.smoothB;}
    this.updateUniforms();pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:.005,g:.008,b:.016,a:1},loadOp:"clear",storeOp:"store"}]});pass.setPipeline(this.compositePipeline);pass.setBindGroup(0,this.bind(this.compositePipeline,current,this.thicknessTexture));pass.draw(3);pass.end();
    this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();pushTiming(this.renderTimes,performance.now()-start);
  }
  report(){return{status:this.failed?"failed":"ready",error:this.failed||"",upload_median_ms:percentile(this.uploadTimes,.5),upload_p95_ms:percentile(this.uploadTimes,.95),render_median_ms:percentile(this.renderTimes,.5),render_p95_ms:percentile(this.renderTimes,.95),dropped_frames:this.dropped,stale_frames:0};}
}

async function renderer() {
  const canvas=await findCanvas();if(renderers.has(canvas))return renderers.get(canvas);
  const instance=new SurfaceRenderer(canvas);renderers.set(canvas,instance);await instance.initialize();return instance;
}

export async function updateSurfaceFrame(frame) {
  if (!frame) return {status:true,outputs:{status:"waiting"}};
  try {const floats=decodeSurfaceFrame(frame);const instance=await renderer();await instance.setFrame(frame,floats);return {status:true,outputs:instance.report()};}
  catch(error){return{status:false,outputs:{status:"unavailable",error:String(error?.message||error)}};}
}

export async function updateSurfaceControls(controls) {
  deferredControls={...deferredControls,...(controls||{})};
  for(const instance of renderers.values()){instance.controls={...instance.controls,...deferredControls};instance.scheduleRender();}
  return {status:true,outputs:{status:"ready"}};
}

export async function resetSurfaceCamera() {
  for(const instance of renderers.values())instance.resetCamera();
  return {status:true,outputs:{status:"ready"}};
}
