import assert from "node:assert/strict";
import test from "node:test";

globalThis.atob = (value) => Buffer.from(value, "base64").toString("binary");

const module = await import("./assets/webgpu_geospatial.js");

function encoded(values) {
  return Buffer.from(new Uint16Array(values).buffer).toString("base64");
}

test("terrain decoder expands little-endian u16 quantization", () => {
  const frame={version:1,kind:"terrain",shape:[2,3],minimum:10,scale:.5,data:encoded([0,2,4,6,8,10])};
  const decoded=module.decodeGeospatialTerrain(frame);
  assert.equal(decoded.nx,3);assert.equal(decoded.ny,2);
  assert.deepEqual(Array.from(decoded.values),[10,11,12,13,14,15]);
});

test("water decoder expands interleaved depth and speed", () => {
  const frame={version:1,kind:"water",shape:[2,2],depth_minimum:0,depth_scale:.1,speed_minimum:1,speed_scale:.25,data:encoded([10,0,20,2,30,4,40,6])};
  const decoded=module.decodeGeospatialFrame(frame);
  assert.deepEqual(Array.from(decoded.values),[1,1,2,1.5,3,2,4,2.5]);
});

test("decoders reject malformed version and lengths", () => {
  assert.throws(()=>module.decodeGeospatialTerrain({version:2,kind:"terrain"}),/version/);
  assert.throws(()=>module.decodeGeospatialFrame({version:1,kind:"water",shape:[2,2],depth_minimum:0,depth_scale:1,speed_minimum:0,speed_scale:1,data:encoded([1])}),/expected/);
});

test("matrix and canvas helpers remain bounded", () => {
  const projection=module.perspectiveZO(Math.PI/4,16/9,.1,10000);
  const view=module.lookAt([100,200,300],[0,0,0]);
  const combined=module.multiplyMat4(projection,view);
  assert.equal(combined.length,16);assert.ok(Array.from(combined).every(Number.isFinite));
  assert.deepEqual(module.canvasPixelSize(100.9,50.9,3),[201,101]);
});

test("shader contains terrain, water, wet-front and storage contracts", () => {
  const shader=module.GEOSPATIAL_WGSL;
  for(const token of ["vs_terrain","vs_water","fs_terrain","fs_water","water_state","shoreline","discard"]){assert.ok(shader.includes(token),token);}
});
