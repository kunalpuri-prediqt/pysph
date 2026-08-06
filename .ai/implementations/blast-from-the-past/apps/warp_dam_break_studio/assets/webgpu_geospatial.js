/* Client WebGPU heightfield renderer for the geospatial SWE profile. */

export const GEOSPATIAL_FRAME_VERSION = 1;

const renderers = new Map();
let deferredControls = {};

function decodeBase64U16(payload, expectedValues, label) {
  const binary = atob(payload || "");
  if (binary.length !== expectedValues * 2) {
    throw new Error(`${label} payload has ${binary.length} bytes; expected ${expectedValues * 2}`);
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < bytes.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
}

function validShape(frame) {
  const shape = frame?.shape;
  if (!Array.isArray(shape) || shape.length !== 2) throw new Error("Invalid geospatial grid shape");
  const ny = Number(shape[0]); const nx = Number(shape[1]);
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 2 || ny < 2) {
    throw new Error("Invalid geospatial grid dimensions");
  }
  return [ny, nx];
}

export function decodeGeospatialTerrain(frame) {
  if (!frame || frame.version !== GEOSPATIAL_FRAME_VERSION || frame.kind !== "terrain") {
    throw new Error("Unsupported or missing geospatial terrain version");
  }
  const [ny, nx] = validShape(frame);
  const encoded = decodeBase64U16(frame.data, nx * ny, "Terrain");
  const minimum = Number(frame.minimum); const scale = Number(frame.scale);
  if (!Number.isFinite(minimum) || !Number.isFinite(scale) || scale <= 0) {
    throw new Error("Invalid terrain quantization metadata");
  }
  const values = new Float32Array(encoded.length);
  for (let i = 0; i < encoded.length; i += 1) values[i] = minimum + encoded[i] * scale;
  return {values, nx, ny};
}

export function decodeGeospatialFrame(frame) {
  if (!frame || frame.version !== GEOSPATIAL_FRAME_VERSION || frame.kind !== "water") {
    throw new Error("Unsupported or missing geospatial water version");
  }
  const [ny, nx] = validShape(frame);
  const encoded = decodeBase64U16(frame.data, nx * ny * 2, "Water");
  const depthMinimum = Number(frame.depth_minimum); const depthScale = Number(frame.depth_scale);
  const speedMinimum = Number(frame.speed_minimum); const speedScale = Number(frame.speed_scale);
  if (![depthMinimum, depthScale, speedMinimum, speedScale].every(Number.isFinite)
      || depthScale <= 0 || speedScale <= 0) {
    throw new Error("Invalid water quantization metadata");
  }
  const values = new Float32Array(nx * ny * 2);
  for (let i = 0; i < nx * ny; i += 1) {
    values[i * 2] = depthMinimum + encoded[i * 2] * depthScale;
    values[i * 2 + 1] = speedMinimum + encoded[i * 2 + 1] * speedScale;
  }
  return {values, nx, ny};
}

export function perspectiveZO(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0, 0, f, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, near * far / (near - far), 0,
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
  const length = Math.hypot(...v) || 1;
  return v.map((value) => value / length);
};

export function lookAt(eye, target, up = [0, 0, 1]) {
  const z = normalize3(sub3(eye, target));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0, x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0, -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
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

export function canvasPixelSize(width, height, ratio = 1) {
  const dpr = Math.min(Math.max(Number(ratio) || 1, 1), 2);
  return [Math.max(1, Math.floor(width * dpr)), Math.max(1, Math.floor(height * dpr))];
}

export const GEOSPATIAL_WGSL = /* wgsl */ `
struct Uniforms {
  view_proj: mat4x4<f32>,
  camera: vec4<f32>,
  grid: vec4<f32>,
  elevation: vec4<f32>,
  light: vec4<f32>,
  water: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> terrain: array<f32>;
@group(0) @binding(2) var<storage, read> water_state: array<vec2<f32>>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) elevation: f32,
  @location(3) depth: f32,
  @location(4) speed: f32,
};

fn index_at(x: u32, y: u32) -> u32 { return y * u32(u.grid.x) + x; }
fn clamped_index(x: i32, y: i32) -> u32 {
  let cx = u32(clamp(x, 0, i32(u.grid.x) - 1));
  let cy = u32(clamp(y, 0, i32(u.grid.y) - 1));
  return index_at(cx, cy);
}
fn corner(vertex: u32) -> vec2<u32> {
  return array<vec2<u32>, 6>(vec2(0,0),vec2(1,0),vec2(0,1),vec2(0,1),vec2(1,0),vec2(1,1))[vertex % 6u];
}
fn grid_xy(vertex: u32) -> vec2<u32> {
  let cell = vertex / 6u; let width = u32(u.grid.x) - 1u;
  return vec2(cell % width, cell / width) + corner(vertex);
}
fn terrain_normal(x: u32, y: u32, include_water: bool) -> vec3<f32> {
  let xi=i32(x); let yi=i32(y);
  let left=clamped_index(xi-1,yi); let right=clamped_index(xi+1,yi);
  let lower=clamped_index(xi,yi-1); let upper=clamped_index(xi,yi+1);
  var zl=terrain[left]; var zr=terrain[right]; var zd=terrain[lower]; var zu=terrain[upper];
  if(include_water){zl+=water_state[left].x;zr+=water_state[right].x;zd+=water_state[lower].x;zu+=water_state[upper].x;}
  let dzdx=(zr-zl)*u.elevation.z/(2.0*u.grid.z);
  let dzdy=(zu-zd)*u.elevation.z/(2.0*u.grid.w);
  return normalize(vec3(-dzdx,-dzdy,1.0));
}
fn make_vertex(vertex: u32, is_water: bool) -> VertexOut {
  let xy=grid_xy(vertex); let index=index_at(xy.x,xy.y);
  let bed=terrain[index]; let state=water_state[index];
  let z=(bed-u.elevation.x + select(0.0,state.x,is_water))*u.elevation.z;
  let width=(u.grid.x-1.0)*u.grid.z; let height=(u.grid.y-1.0)*u.grid.w;
  let world=vec3(f32(xy.x)*u.grid.z-width*0.5,f32(xy.y)*u.grid.w-height*0.5,z);
  var out:VertexOut; out.position=u.view_proj*vec4(world,1.0); out.world=world;
  out.normal=terrain_normal(xy.x,xy.y,is_water); out.elevation=bed; out.depth=state.x; out.speed=state.y; return out;
}
@vertex fn vs_terrain(@builtin(vertex_index) vertex:u32)->VertexOut{return make_vertex(vertex,false);}
@vertex fn vs_water(@builtin(vertex_index) vertex:u32)->VertexOut{return make_vertex(vertex,true);}

fn sun_light(normal:vec3<f32>, world:vec3<f32>)->f32{
  let diffuse=max(dot(normal,normalize(u.light.xyz)),0.0);
  return 0.24+0.76*diffuse;
}
@fragment fn fs_terrain(in:VertexOut)->@location(0) vec4<f32>{
  let height=clamp((in.elevation-u.elevation.x)/max(u.elevation.y,0.001),0.0,1.0);
  let slope=1.0-clamp(in.normal.z,0.0,1.0);
  let valley=mix(vec3(0.055,0.13,0.075),vec3(0.16,0.25,0.105),smoothstep(0.0,0.54,height));
  let rock=mix(vec3(0.22,0.21,0.18),vec3(0.42,0.39,0.32),height);
  let base=mix(valley,rock,smoothstep(0.25,0.72,slope+height*0.30));
  let contour=0.91+0.09*smoothstep(0.08,0.46,abs(fract(in.elevation/12.0)-0.5));
  let fog=clamp(length(in.world-u.camera.xyz)/(u.elevation.w*2.55),0.0,0.46);
  let color=base*sun_light(in.normal,in.world)*contour;
  return vec4(mix(color,vec3(0.17,0.28,0.34),fog),1.0);
}
@fragment fn fs_water(in:VertexOut)->@location(0) vec4<f32>{
  if(in.depth<u.water.w){discard;}
  let view=normalize(u.camera.xyz-in.world); let ndv=max(dot(in.normal,view),0.0);
  let fresnel=0.025+0.26*pow(1.0-ndv,5.0);
  let shallow=vec3(0.035,0.38,0.48); let deep=vec3(0.008,0.075,0.14);
  var color=mix(shallow,deep,1.0-exp(-in.depth*u.water.x));
  let halfv=normalize(normalize(u.light.xyz)+view);
  let highlight=pow(max(dot(in.normal,halfv),0.0),180.0)*0.16;
  let shoreline=1.0-smoothstep(u.water.w*1.5,u.water.w*8.0,in.depth);
  let turbulence=smoothstep(u.water.y,u.water.z,in.speed);
  let foam=clamp(shoreline*0.42+turbulence*0.46,0.0,0.64);
  color=mix(color,vec3(0.63,0.79,0.80),foam);
  color=mix(color,vec3(0.18,0.34,0.43),fresnel)+highlight;
  return vec4(color,0.90);
}`;

const pushTiming = (array, value) => { array.push(value); if (array.length > 180) array.shift(); };
const percentile = (values, p) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
};

async function findCanvas() {
  for (let i = 0; i < 60; i += 1) {
    const canvas = document.getElementById("geospatial-canvas");
    if (canvas) return canvas;
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  throw new Error("Geospatial WebGPU canvas was not mounted");
}

class GeospatialRenderer {
  constructor(canvas) {
    this.canvas=canvas; this.controls={...deferredControls}; this.terrain=null; this.frame=null;
    this.uploadTimes=[]; this.renderTimes=[]; this.pointer=null; this.needsRender=false;this.pendingResolvers=[];
  }
  async initialize() {
    if (!navigator.gpu) throw new Error("This browser does not expose WebGPU");
    this.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
    if(!this.adapter)throw new Error("No WebGPU adapter is available");
    this.device=await this.adapter.requestDevice(); this.context=this.canvas.getContext("webgpu");
    this.format=navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"});
    this.module=this.device.createShaderModule({label:"Geospatial SWE WGSL",code:GEOSPATIAL_WGSL});
    const compilation=await this.module.getCompilationInfo();
    const errors=compilation.messages.filter((message)=>message.type==="error");
    if(errors.length)throw new Error(errors.map((message)=>`WGSL ${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));
    this.uniformBuffer=this.device.createBuffer({size:144,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.terrainBuffer=this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.waterBuffer=this.device.createBuffer({size:8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.terrainPipeline=await this.device.createRenderPipelineAsync({layout:"auto",vertex:{module:this.module,entryPoint:"vs_terrain"},fragment:{module:this.module,entryPoint:"fs_terrain",targets:[{format:this.format}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
    this.waterPipeline=await this.device.createRenderPipelineAsync({layout:"auto",vertex:{module:this.module,entryPoint:"vs_water"},fragment:{module:this.module,entryPoint:"fs_water",targets:[{format:this.format,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less-equal"}});
    this.installCamera(); this.resizeObserver=new ResizeObserver(()=>this.scheduleRender()); this.resizeObserver.observe(this.canvas);
  }
  installCamera(){
    this.canvas.addEventListener("contextmenu",(event)=>event.preventDefault());
    this.canvas.addEventListener("pointerdown",(event)=>{this.pointer={x:event.clientX,y:event.clientY,button:event.button};try{this.canvas.setPointerCapture(event.pointerId);}catch(_error){}});
    this.canvas.addEventListener("pointerup",()=>{this.pointer=null;});
    this.canvas.addEventListener("pointermove",(event)=>{if(!this.pointer)return;const dx=event.clientX-this.pointer.x,dy=event.clientY-this.pointer.y;this.pointer.x=event.clientX;this.pointer.y=event.clientY;if(this.pointer.button===0&&!event.shiftKey){this.camera.yaw-=dx*0.005;this.camera.pitch=Math.max(0.12,Math.min(1.35,this.camera.pitch+dy*0.004));}else{const scale=this.camera.distance*0.0015;this.camera.target[0]-=dx*scale;this.camera.target[2]+=dy*scale;}this.scheduleRender();});
    this.canvas.addEventListener("wheel",(event)=>{event.preventDefault();this.camera.distance=Math.max(this.domain*0.08,Math.min(this.domain*3.2,this.camera.distance*Math.exp(event.deltaY*0.001)));this.scheduleRender();},{passive:false});
    this.canvas.addEventListener("dblclick",()=>this.resetCamera());
  }
  resetCamera(){
    const domain=this.domain||1000; const relief=this.relief||100;
    this.camera={target:[0,0,relief*0.72],yaw:-1.12,pitch:0.42,distance:domain*0.93}; this.scheduleRender();
  }
  scheduleRender(){if(this.needsRender)return;this.needsRender=true;requestAnimationFrame(()=>{this.needsRender=false;if(this.terrain&&this.frame)this.render();});}
  ensureSize(){
    const [width,height]=canvasPixelSize(this.canvas.clientWidth,this.canvas.clientHeight,window.devicePixelRatio);
    if(this.canvas.width===width&&this.canvas.height===height&&this.depthTexture)return;
    this.canvas.width=width;this.canvas.height=height;if(this.depthTexture)this.depthTexture.destroy();
    this.depthTexture=this.device.createTexture({size:[width,height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});
  }
  replaceBuffer(name,values){if(this[name])this[name].destroy();this[name]=this.device.createBuffer({size:Math.max(4,values.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(this[name],0,values);}
  async setTerrain(frame,decoded){
    this.terrain=frame;this.nx=decoded.nx;this.ny=decoded.ny;this.terrainValues=decoded.values;
    this.minimum=Infinity;this.maximum=-Infinity;
    for(const value of decoded.values){this.minimum=Math.min(this.minimum,value);this.maximum=Math.max(this.maximum,value);}
    this.relief=Math.max(1,this.maximum-this.minimum);
    const cell=frame.cell_size||[30,30];this.dx=Number(cell[0]);this.dy=Number(cell[1]);this.domain=Math.max((this.nx-1)*this.dx,(this.ny-1)*this.dy);
    this.replaceBuffer("terrainBuffer",decoded.values);this.resetCamera();
    if(this.pendingFrame){const pending=this.pendingFrame;this.pendingFrame=null;await this.setFrame(pending.frame,pending.decoded);for(const resolve of this.pendingResolvers.splice(0))resolve(true);}
  }
  async setFrame(frame,decoded){
    if(!this.terrain){this.pendingFrame={frame,decoded};return new Promise((resolve)=>this.pendingResolvers.push(resolve));}
    if(decoded.nx!==this.nx||decoded.ny!==this.ny)throw new Error("Terrain/water grid mismatch");
    this.frame=frame;const start=performance.now();this.replaceBuffer("waterBuffer",decoded.values);pushTiming(this.uploadTimes,performance.now()-start);await this.render();return true;
  }
  updateUniforms(){
    this.ensureSize();const c=this.camera;const cp=Math.cos(c.pitch);const eye=[c.target[0]+c.distance*cp*Math.cos(c.yaw),c.target[1]+c.distance*cp*Math.sin(c.yaw),c.target[2]+c.distance*Math.sin(c.pitch)];
    const view=lookAt(eye,c.target),projection=perspectiveZO(Math.PI/4,this.canvas.width/this.canvas.height,Math.max(0.5,this.domain*0.0002),this.domain*6),vp=multiplyMat4(projection,view);
    const values=new Float32Array(36);values.set(vp,0);values.set([...eye,1],16);values.set([this.nx,this.ny,this.dx,this.dy],20);
    const exaggeration=Number(this.controls.vertical_exaggeration??4.5);values.set([this.minimum,this.relief,exaggeration,this.domain],24);values.set([-0.38,-0.48,0.79,0],28);
    values.set([Number(this.controls.absorption??0.08),Number(this.controls.foam_start??2.0),Number(this.controls.foam_end??8.0),Number(this.controls.dry_depth??0.01)],32);
    this.device.queue.writeBuffer(this.uniformBuffer,0,values);
  }
  bind(pipeline){return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.terrainBuffer}},{binding:2,resource:{buffer:this.waterBuffer}}]});}
  async render(){
    if(!this.terrain||!this.frame)return;this.updateUniforms();const start=performance.now();const encoder=this.device.createCommandEncoder({label:"Geospatial terrain"});
    const color=this.context.getCurrentTexture().createView();const depth=this.depthTexture.createView();
    let pass=encoder.beginRenderPass({colorAttachments:[{view:color,clearValue:{r:.17,g:.29,b:.38,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});
    const vertices=(this.nx-1)*(this.ny-1)*6;pass.setPipeline(this.terrainPipeline);pass.setBindGroup(0,this.bind(this.terrainPipeline));pass.draw(vertices);pass.end();
    pass=encoder.beginRenderPass({colorAttachments:[{view:color,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:depth,depthLoadOp:"load",depthStoreOp:"store"}});pass.setPipeline(this.waterPipeline);pass.setBindGroup(0,this.bind(this.waterPipeline));pass.draw(vertices);pass.end();
    this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();pushTiming(this.renderTimes,performance.now()-start);
  }
  report(){return{status:"ready",upload_median_ms:percentile(this.uploadTimes,.5),upload_p95_ms:percentile(this.uploadTimes,.95),render_median_ms:percentile(this.renderTimes,.5),render_p95_ms:percentile(this.renderTimes,.95)};}
}

async function renderer(){const canvas=await findCanvas();if(renderers.has(canvas))return renderers.get(canvas);const instance=new GeospatialRenderer(canvas);renderers.set(canvas,instance);await instance.initialize();return instance;}

export async function updateGeospatialTerrain(frame){
  if(!frame)return{status:true,outputs:{status:"waiting"}};
  try{const decoded=decodeGeospatialTerrain(frame);const instance=await renderer();await instance.setTerrain(frame,decoded);return{status:true,outputs:instance.report()};}
  catch(error){return{status:false,outputs:{status:"unavailable",error:String(error?.message||error)}};}
}
export async function updateGeospatialFrame(frame){
  if(!frame)return{status:true,outputs:{status:"waiting"}};
  try{const decoded=decodeGeospatialFrame(frame);const instance=await renderer();const rendered=await instance.setFrame(frame,decoded);return{status:true,outputs:rendered?instance.report():{status:"waiting"}};}
  catch(error){return{status:false,outputs:{status:"unavailable",error:String(error?.message||error)}};}
}
export async function updateGeospatialControls(controls){deferredControls={...deferredControls,...(controls||{})};for(const instance of renderers.values()){instance.controls={...instance.controls,...deferredControls};instance.scheduleRender();}return{status:true,outputs:{status:"ready"}};}
export async function resetGeospatialCamera(){for(const instance of renderers.values())instance.resetCamera();return{status:true,outputs:{status:"ready"}};}
