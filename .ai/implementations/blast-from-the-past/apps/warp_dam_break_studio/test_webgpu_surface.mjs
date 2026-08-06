import assert from "node:assert/strict";
import test from "node:test";

import {
  SURFACE_FRAME_VERSION,
  SURFACE_STRIDE_FLOATS,
  SURFACE_WGSL,
  canvasPixelSize,
  decodeSurfaceFrame,
  invertMat4,
  lookAt,
  multiplyMat4,
  perspectiveZO,
  webGPUAvailable,
} from "./assets/webgpu_surface.js";

const encodedFrame = (values, count) => ({
  version: SURFACE_FRAME_VERSION,
  count,
  stride_floats: SURFACE_STRIDE_FLOATS,
  payload: Buffer.from(new Float32Array(values).buffer).toString("base64"),
});

test("surface frame decoder preserves exact float32 values", () => {
  const values = Array.from({length: 32}, (_, index) => index * 0.25);
  assert.deepEqual(Array.from(decodeSurfaceFrame(encodedFrame(values, 2))), values);
});

test("surface frame decoder rejects header and length mismatches", () => {
  assert.throws(
    () => decodeSurfaceFrame({...encodedFrame([], 0), version: 99}),
    /version/i,
  );
  assert.throws(() => decodeSurfaceFrame(encodedFrame([1, 2], 1)), /expected/i);
});

test("camera matrices invert to identity", () => {
  const projection = perspectiveZO(Math.PI / 4, 16 / 9, 0.02, 30);
  const view = lookAt([3, -5, 2], [2.4, 0, 0.5]);
  const matrix = multiplyMat4(projection, view);
  const identity = multiplyMat4(matrix, invertMat4(matrix));
  for (let index = 0; index < 16; index += 1) {
    const expected = index % 5 === 0 ? 1 : 0;
    assert.ok(Math.abs(identity[index] - expected) < 2e-5);
  }
});

test("canvas sizing is bounded and deterministic", () => {
  assert.deepEqual(canvasPixelSize(640, 360, 1.5), [960, 540]);
  assert.deepEqual(canvasPixelSize(640, 360, 4), [1280, 720]);
  assert.deepEqual(canvasPixelSize(0, 0, 1), [1, 1]);
});

test("capability check fails cleanly without navigator.gpu", () => {
  assert.equal(webGPUAvailable({}), false);
  assert.equal(webGPUAvailable({navigator: {}}), false);
  assert.equal(webGPUAvailable({navigator: {gpu: {}}}), true);
});

test("shader contains every multipass contract", () => {
  for (const entry of [
    "vs_splat", "fs_depth", "fs_thickness", "fs_smooth", "fs_composite",
  ]) {
    assert.match(SURFACE_WGSL, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(SURFACE_WGSL, /@builtin\(frag_depth\)/);
  assert.match(SURFACE_WGSL, /fresnel/i);
  assert.match(SURFACE_WGSL, /absorption/i);
  assert.match(SURFACE_WGSL, /delta\*delta \* 0\.75/);
  assert.match(SURFACE_WGSL, /fn filtered_thickness\b/);
  assert.match(SURFACE_WGSL, /phase\*2\.1/);
  assert.doesNotMatch(SURFACE_WGSL, /\.[xyzwrgba]{2,}\s*[+*/-]=/);
});
