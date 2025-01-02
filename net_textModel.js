
const textModel = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE });
  new Uint8Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_386_32_12352_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 386 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+(gidx0<<5));
  var alu1 = (alu0+alu0+(gidx0<<6)+(lidx0<<1));
  var alu2 = ((gidx0<<7)+(lidx0<<2));
  data0[alu2] = (f32(alu1));
  data0[(alu2+1)] = (f32((alu1+1)));
  data0[(alu2+2)] = (f32((alu1+2)));
  data0[(alu2+3)] = (f32((alu1+3)));
}`;

const r_77_77n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<i32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  data0[gidx0] = gidx0;
}`;

const E_77_77 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 77 */
  data0[(gidx0+(gidx1*77))] = select(0.0f,(f32(-INFINITY)),(((gidx0+(gidx1*305)+305)%306)<153));
}`;

const r_77_32_4_8_16_193_3_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 77 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[gidx2];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 193; ridx0++) {
    var alu0 = ((gidx0*9486336)+(gidx1*24)+(lidx0*3)+(lidx1*592896)+(ridx0*3072));
    var val1 = data3[alu0];
    var val2 = data3[(alu0+1)];
    var val3 = data3[(alu0+2)];
    var val4 = data3[(alu0+768)];
    var val5 = data3[(alu0+769)];
    var val6 = data3[(alu0+770)];
    var val7 = data3[(alu0+1536)];
    var val8 = data3[(alu0+1537)];
    var val9 = data3[(alu0+1538)];
    var val10 = data3[(alu0+2304)];
    var val11 = data3[(alu0+2305)];
    var val12 = data3[(alu0+2306)];
    var alu1 = ((gidx0*12352)+(lidx1*772)+(ridx0<<2));
    var val13 = data1[alu1];
    var val14 = data1[(alu1+1)];
    var val15 = data1[(alu1+2)];
    var val16 = data1[(alu1+3)];
    var cast0 = (f32(((val14!=val0)!=true)));
    var cast1 = (f32(((val15!=val0)!=true)));
    var cast2 = (f32(((val16!=val0)!=true)));
    var cast3 = (f32(((val13!=val0)!=true)));
    acc0 = (acc0+(cast0*val4)+(cast3*val1)+(cast1*val7)+(cast2*val10));
    acc1 = (acc1+(cast0*val5)+(cast3*val2)+(cast1*val8)+(cast2*val11));
    acc2 = (acc2+(cast0*val6)+(cast3*val3)+(cast1*val9)+(cast2*val12));
  }
  var alu6 = (lidx1+(gidx1*1536)+(gidx2*49152)+(gidx0<<4)+(lidx0*192));
  data0[alu6] = acc0;
  data0[(alu6+64)] = acc1;
  data0[(alu6+128)] = acc2;
}`;

const r_77_16_16_77_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2:array<i32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0*48);
  var alu1 = (lidx0*3);
  var val0 = data2[gidx1];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val1 = data1[ridx0];
    var alu2 = (alu0+alu1+(ridx0*768));
    var val2 = data3[alu2];
    var val3 = data3[(alu2+1)];
    var val4 = data3[(alu2+2)];
    var cast0 = (f32(((val1!=val0)!=true)));
    acc0 = (acc0+(cast0*val2));
    acc1 = (acc1+(cast0*val3));
    acc2 = (acc2+(cast0*val4));
  }
  var alu7 = (alu0+(gidx1*768)+alu1);
  data0[alu7] = acc0;
  data0[(alu7+1)] = acc1;
  data0[(alu7+2)] = acc2;
}`;

const r_1848_32_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1848 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var alu0 = ((gidx0<<11)+(lidx0<<6)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val3+val2+val1+val0);
  }
  var alu3 = (lidx0+(gidx0<<5));
  var val4 = data2[alu3];
  data0[alu3] = (val4+acc0);
}`;

const r_77_16_48 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 48; ridx0++) {
    var val0 = data1[((gidx0*768)+(lidx0*48)+ridx0)];
    acc0 = (acc0+val0);
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (acc1*0.0013020833721384406f);
  }
}`;

const r_77_16_48n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 48; ridx0++) {
    var val1 = data1[((gidx0*768)+(lidx0*48)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(alu0*alu0));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((acc1*0.0013020833721384406f)+1e-05f)));
  }
}`;

const E_77_16_16_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx1];
  var val1 = data3[gidx1];
  var alu0 = (gidx0*48);
  var alu1 = (lidx0*3);
  var alu2 = (alu0+(gidx1*768)+alu1);
  var val2 = data1[alu2];
  var alu3 = (alu2+1);
  var val3 = data1[alu3];
  var alu4 = (alu2+2);
  var val4 = data1[alu4];
  var alu5 = (alu0+alu1);
  var val5 = data4[alu5];
  var val6 = data5[alu5];
  var alu6 = (alu5+1);
  var val7 = data4[alu6];
  var val8 = data5[alu6];
  var alu7 = (alu5+2);
  var val9 = data4[alu7];
  var val10 = data5[alu7];
  data0[alu3] = (val8+(val7*val1*(val3-val0)));
  data0[alu4] = (val10+(val9*val1*(val4-val0)));
  data0[alu2] = (val6+(val5*val1*(val2-val0)));
}`;

const r_77_16_16_192_3_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx1*768);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 192; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*36864)+(lidx0*2304)+alu1);
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+768)];
    var val5 = data2[(alu2+769)];
    var val6 = data2[(alu2+770)];
    var val7 = data2[(alu2+771)];
    var val8 = data2[(alu2+1536)];
    var val9 = data2[(alu2+1537)];
    var val10 = data2[(alu2+1538)];
    var val11 = data2[(alu2+1539)];
    var alu3 = (alu0+alu1);
    var val12 = data1[alu3];
    var val13 = data1[(alu3+1)];
    var val14 = data1[(alu3+2)];
    var val15 = data1[(alu3+3)];
    acc0 = (acc0+(val13*val1)+(val12*val0)+(val14*val2)+(val15*val3));
    acc1 = (acc1+(val13*val5)+(val12*val4)+(val14*val6)+(val15*val7));
    acc2 = (acc2+(val13*val9)+(val12*val8)+(val14*val10)+(val15*val11));
  }
  var alu8 = (gidx0*48);
  var alu9 = (lidx0*3);
  var alu10 = (alu8+alu9);
  var val16 = data3[alu10];
  var val17 = data3[(alu10+1)];
  var val18 = data3[(alu10+2)];
  var alu11 = (alu8+alu0+alu9);
  data0[(alu11+1)] = (val17+acc1);
  data0[(alu11+2)] = (val18+acc2);
  data0[alu11] = (val16+acc0);
}`;

const r_77_77_4_16_3_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 4 */
  var alu0 = (lidx0*192);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*768)+alu0+alu1);
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+64)];
    var val5 = data2[(alu2+65)];
    var val6 = data2[(alu2+66)];
    var val7 = data2[(alu2+67)];
    var val8 = data2[(alu2+128)];
    var val9 = data2[(alu2+129)];
    var val10 = data2[(alu2+130)];
    var val11 = data2[(alu2+131)];
    var alu3 = ((gidx1*768)+alu0+alu1);
    var val12 = data1[alu3];
    var val13 = data1[(alu3+1)];
    var val14 = data1[(alu3+2)];
    var val15 = data1[(alu3+3)];
    var val16 = data1[(alu3+64)];
    var val17 = data1[(alu3+65)];
    var val18 = data1[(alu3+66)];
    var val19 = data1[(alu3+67)];
    var val20 = data1[(alu3+128)];
    var val21 = data1[(alu3+129)];
    var val22 = data1[(alu3+130)];
    var val23 = data1[(alu3+131)];
    acc0 = (acc0+(val13*val1)+(val12*val0)+(val14*val2)+(val15*val3));
    acc1 = (acc1+(val16*val4)+(val17*val5)+(val18*val6)+(val19*val7));
    acc2 = (acc2+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
  }
  var alu8 = (gidx0+(gidx1*77));
  var val24 = data3[alu8];
  var alu9 = (alu8+(lidx0*17787));
  data0[alu9] = (val24+(acc0*0.125f));
  data0[(alu9+5929)] = (val24+(acc1*0.125f));
  data0[(alu9+11858)] = (val24+(acc2*0.125f));
}`;

const r_231_4_77 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 231 */
  var lidx0 = i32(lindex.x); /* 4 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val0 = data1[((gidx0*308)+(lidx0*77)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  data0[(lidx0+(gidx0<<2))] = acc0;
}`;

const r_231_4_77n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 231 */
  var lidx0 = i32(lindex.x); /* 4 */
  var alu0 = (lidx0+(gidx0<<2));
  var val0 = data2[alu0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val1 = data1[((gidx0*308)+(lidx0*77)+ridx0)];
    acc0 = (acc0+exp2(((val1-val0)*1.4426950408889634f)));
  }
  data0[alu0] = acc0;
}`;

const E_77_77_4_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 4 */
  var alu0 = ((gidx1*12)+(lidx0*3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var alu1 = (alu0+1);
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = (alu0+2);
  var val4 = data2[alu2];
  var val5 = data3[alu2];
  var alu3 = (gidx0+(gidx1*924)+(lidx0*231));
  var val6 = data1[alu3];
  var alu4 = (alu3+77);
  var val7 = data1[alu4];
  var alu5 = (alu3+154);
  var val8 = data1[alu5];
  data0[alu3] = (exp2(((val6-val0)*1.4426950408889634f))*(1/val1));
  data0[alu4] = (exp2(((val7-val2)*1.4426950408889634f))*(1/val3));
  data0[alu5] = (exp2(((val8-val4)*1.4426950408889634f))*(1/val5));
}`;

const r_3_77_4_16_77_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 3 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx1<<2);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val0 = data1[((gidx0*77)+(gidx1*23716)+(lidx0*5929)+ridx0)];
    var alu1 = ((gidx1<<8)+(lidx0<<6)+alu0+(ridx0*768));
    var val1 = data2[alu1];
    var val2 = data2[(alu1+1)];
    var val3 = data2[(alu1+2)];
    var val4 = data2[(alu1+3)];
    acc0 = (acc0+(val0*val1));
    acc1 = (acc1+(val0*val2));
    acc2 = (acc2+(val0*val3));
    acc3 = (acc3+(val0*val4));
  }
  var alu7 = ((gidx0<<6)+(gidx1*19712)+(lidx0*4928)+alu0);
  data0[alu7] = acc0;
  data0[(alu7+1)] = acc1;
  data0[(alu7+2)] = acc2;
  data0[(alu7+3)] = acc3;
}`;

const r_77_16_16_192_3_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 192; ridx0++) {
    var alu0 = ((gidx0*36864)+(lidx0*2304)+(ridx0<<2));
    var val0 = data3[alu0];
    var val1 = data3[(alu0+1)];
    var val2 = data3[(alu0+2)];
    var val3 = data3[(alu0+3)];
    var val4 = data3[(alu0+768)];
    var val5 = data3[(alu0+769)];
    var val6 = data3[(alu0+770)];
    var val7 = data3[(alu0+771)];
    var val8 = data3[(alu0+1536)];
    var val9 = data3[(alu0+1537)];
    var val10 = data3[(alu0+1538)];
    var val11 = data3[(alu0+1539)];
    var alu1 = ((gidx1<<6)+((ridx0>>4)*4928)+((ridx0&15)<<2));
    var val12 = data2[alu1];
    var val13 = data2[(alu1+1)];
    var val14 = data2[(alu1+2)];
    var val15 = data2[(alu1+3)];
    acc0 = (acc0+(val13*val1)+(val12*val0)+(val14*val2)+(val15*val3));
    acc1 = (acc1+(val13*val5)+(val12*val4)+(val14*val6)+(val15*val7));
    acc2 = (acc2+(val13*val9)+(val12*val8)+(val14*val10)+(val15*val11));
  }
  var alu6 = (gidx0*48);
  var alu7 = (lidx0*3);
  var alu8 = (alu6+(gidx1*768)+alu7);
  var val16 = data1[alu8];
  var alu9 = (alu8+1);
  var val17 = data1[alu9];
  var alu10 = (alu8+2);
  var val18 = data1[alu10];
  var alu11 = (alu6+alu7);
  var val19 = data4[alu11];
  var val20 = data4[(alu11+1)];
  var val21 = data4[(alu11+2)];
  data0[alu9] = (val17+val20+acc1);
  data0[alu10] = (val18+val21+acc2);
  data0[alu8] = (val16+val19+acc0);
}`;

const r_77_64_16_192_3_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 192; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*36864)+(lidx0*2304)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+768)];
    var val5 = data2[(alu1+769)];
    var val6 = data2[(alu1+770)];
    var val7 = data2[(alu1+771)];
    var val8 = data2[(alu1+1536)];
    var val9 = data2[(alu1+1537)];
    var val10 = data2[(alu1+1538)];
    var val11 = data2[(alu1+1539)];
    var alu2 = ((gidx1*768)+alu0);
    var val12 = data1[alu2];
    var val13 = data1[(alu2+1)];
    var val14 = data1[(alu2+2)];
    var val15 = data1[(alu2+3)];
    acc0 = (acc0+(val13*val1)+(val12*val0)+(val14*val2)+(val15*val3));
    acc1 = (acc1+(val13*val5)+(val12*val4)+(val14*val6)+(val15*val7));
    acc2 = (acc2+(val13*val9)+(val12*val8)+(val14*val10)+(val15*val11));
  }
  var alu7 = (gidx0*48);
  var alu8 = (lidx0*3);
  var alu9 = (alu7+alu8);
  var val16 = data3[alu9];
  var val17 = data3[(alu9+1)];
  var val18 = data3[(alu9+2)];
  var alu10 = (val17+acc1);
  var alu11 = (val18+acc2);
  var alu12 = (val16+acc0);
  var alu13 = (alu7+(gidx1*3072)+alu8);
  data0[(alu13+1)] = ((1/(exp2((alu10*-2.4554669595930156f))+1.0f))*alu10);
  data0[(alu13+2)] = ((1/(exp2((alu11*-2.4554669595930156f))+1.0f))*alu11);
  data0[alu13] = ((1/(exp2((alu12*-2.4554669595930156f))+1.0f))*alu12);
}`;

const r_77_16_16_768_3_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  for (var ridx0 = 0; ridx0 < 768; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*147456)+(lidx0*9216)+alu0);
    var val0 = data3[alu1];
    var val1 = data3[(alu1+1)];
    var val2 = data3[(alu1+2)];
    var val3 = data3[(alu1+3)];
    var val4 = data3[(alu1+3072)];
    var val5 = data3[(alu1+3073)];
    var val6 = data3[(alu1+3074)];
    var val7 = data3[(alu1+3075)];
    var val8 = data3[(alu1+6144)];
    var val9 = data3[(alu1+6145)];
    var val10 = data3[(alu1+6146)];
    var val11 = data3[(alu1+6147)];
    var alu2 = ((gidx1*3072)+alu0);
    var val12 = data2[alu2];
    var val13 = data2[(alu2+1)];
    var val14 = data2[(alu2+2)];
    var val15 = data2[(alu2+3)];
    acc0 = (acc0+(val13*val1)+(val12*val0)+(val14*val2)+(val15*val3));
    acc1 = (acc1+(val13*val5)+(val12*val4)+(val14*val6)+(val15*val7));
    acc2 = (acc2+(val13*val9)+(val12*val8)+(val14*val10)+(val15*val11));
  }
  var alu7 = (gidx0*48);
  var alu8 = (lidx0*3);
  var alu9 = (alu7+(gidx1*768)+alu8);
  var val16 = data1[alu9];
  var alu10 = (alu9+1);
  var val17 = data1[alu10];
  var alu11 = (alu9+2);
  var val18 = data1[alu11];
  var alu12 = (alu7+alu8);
  var val19 = data4[alu12];
  var val20 = data4[(alu12+1)];
  var val21 = data4[(alu12+2)];
  data0[alu10] = (val17+val20+acc1);
  data0[alu11] = (val18+val21+acc2);
  data0[alu9] = (val16+val19+acc0);
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 197632);;
    const buf_1 = createEmptyBuf(device, 308);;
    const buf_2 = createEmptyBuf(device, 23716);;
    const buf_3 = createEmptyBuf(device, 15138816);;
    const input0 = createEmptyBuf(device, 308);;
    const buf_4 = createWeightBuf(device, 151781376, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']));
    const buf_5 = createEmptyBuf(device, 236544);;
    const buf_6 = createWeightBuf(device, 308, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.embeddings.position_embedding.arange']));
    const buf_7 = createWeightBuf(device, 236544, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight']));
    const buf_8 = createEmptyBuf(device, 236544);;
    const buf_9 = createEmptyBuf(device, 308);;
    const buf_10 = createEmptyBuf(device, 308);;
    const buf_11 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight']));
    const buf_12 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias']));
    const buf_13 = createEmptyBuf(device, 236544);;
    const buf_14 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight']));
    const buf_15 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias']));
    const buf_16 = createEmptyBuf(device, 236544);;
    const buf_17 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight']));
    const buf_18 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias']));
    const buf_19 = createEmptyBuf(device, 236544);;
    const buf_20 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight']));
    const buf_21 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias']));
    const buf_22 = createEmptyBuf(device, 284592);;
    const buf_23 = createEmptyBuf(device, 3696);;
    const buf_24 = createEmptyBuf(device, 3696);;
    const buf_25 = createEmptyBuf(device, 284592);;
    const buf_26 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight']));
    const buf_27 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias']));
    const buf_28 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.weight']));
    const buf_29 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.bias']));
    const buf_30 = createEmptyBuf(device, 946176);;
    const buf_31 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight']));
    const buf_32 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.bias']));
    const buf_33 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight']));
    const buf_34 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.bias']));
    const buf_35 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.weight']));
    const buf_36 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.bias']));
    const buf_37 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.weight']));
    const buf_38 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.bias']));
    const buf_39 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.weight']));
    const buf_40 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.bias']));
    const buf_41 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.weight']));
    const buf_42 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.bias']));
    const buf_43 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.weight']));
    const buf_44 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.bias']));
    const buf_45 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.weight']));
    const buf_46 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.bias']));
    const buf_47 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.weight']));
    const buf_48 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.bias']));
    const buf_49 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight']));
    const buf_50 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.bias']));
    const buf_51 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.weight']));
    const buf_52 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.bias']));
    const buf_53 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight']));
    const buf_54 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias']));
    const buf_55 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.weight']));
    const buf_56 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.bias']));
    const buf_57 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.weight']));
    const buf_58 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.bias']));
    const buf_59 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.weight']));
    const buf_60 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.bias']));
    const buf_61 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.weight']));
    const buf_62 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.bias']));
    const buf_63 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.weight']));
    const buf_64 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.bias']));
    const buf_65 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.weight']));
    const buf_66 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.bias']));
    const buf_67 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.weight']));
    const buf_68 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.bias']));
    const buf_69 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight']));
    const buf_70 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.bias']));
    const buf_71 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.weight']));
    const buf_72 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.bias']));
    const buf_73 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.weight']));
    const buf_74 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.bias']));
    const buf_75 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.weight']));
    const buf_76 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.bias']));
    const buf_77 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.weight']));
    const buf_78 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.bias']));
    const buf_79 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight']));
    const buf_80 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.bias']));
    const buf_81 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.weight']));
    const buf_82 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.bias']));
    const buf_83 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.weight']));
    const buf_84 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.bias']));
    const buf_85 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.weight']));
    const buf_86 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.bias']));
    const buf_87 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.weight']));
    const buf_88 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.bias']));
    const buf_89 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.weight']));
    const buf_90 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.bias']));
    const buf_91 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.weight']));
    const buf_92 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.bias']));
    const buf_93 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.weight']));
    const buf_94 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.bias']));
    const buf_95 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.weight']));
    const buf_96 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.bias']));
    const buf_97 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.weight']));
    const buf_98 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.bias']));
    const buf_99 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.weight']));
    const buf_100 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.bias']));
    const buf_101 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight']));
    const buf_102 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.bias']));
    const buf_103 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight']));
    const buf_104 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.bias']));
    const buf_105 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.weight']));
    const buf_106 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.bias']));
    const buf_107 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.weight']));
    const buf_108 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.bias']));
    const buf_109 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.weight']));
    const buf_110 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.bias']));
    const buf_111 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.weight']));
    const buf_112 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.bias']));
    const buf_113 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.weight']));
    const buf_114 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.bias']));
    const buf_115 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.weight']));
    const buf_116 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.bias']));
    const buf_117 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.weight']));
    const buf_118 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.bias']));
    const buf_119 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.weight']));
    const buf_120 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.bias']));
    const buf_121 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.weight']));
    const buf_122 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.bias']));
    const buf_123 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.weight']));
    const buf_124 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.bias']));
    const buf_125 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.weight']));
    const buf_126 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.bias']));
    const buf_127 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.weight']));
    const buf_128 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.bias']));
    const buf_129 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.weight']));
    const buf_130 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.bias']));
    const buf_131 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.weight']));
    const buf_132 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.bias']));
    const buf_133 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.weight']));
    const buf_134 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.bias']));
    const buf_135 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.weight']));
    const buf_136 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.bias']));
    const buf_137 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.weight']));
    const buf_138 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.bias']));
    const buf_139 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.weight']));
    const buf_140 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.bias']));
    const buf_141 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.weight']));
    const buf_142 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.bias']));
    const buf_143 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.weight']));
    const buf_144 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.bias']));
    const buf_145 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.weight']));
    const buf_146 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.bias']));
    const buf_147 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.weight']));
    const buf_148 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.bias']));
    const buf_149 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.weight']));
    const buf_150 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.bias']));
    const buf_151 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.weight']));
    const buf_152 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.bias']));
    const buf_153 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.weight']));
    const buf_154 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.bias']));
    const buf_155 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.weight']));
    const buf_156 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.bias']));
    const buf_157 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.weight']));
    const buf_158 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.bias']));
    const buf_159 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.weight']));
    const buf_160 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.bias']));
    const buf_161 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.weight']));
    const buf_162 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.bias']));
    const buf_163 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.weight']));
    const buf_164 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.bias']));
    const buf_165 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.weight']));
    const buf_166 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.bias']));
    const buf_167 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.weight']));
    const buf_168 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.bias']));
    const buf_169 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight']));
    const buf_170 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.bias']));
    const buf_171 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.weight']));
    const buf_172 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.bias']));
    const buf_173 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.weight']));
    const buf_174 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.bias']));
    const buf_175 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.weight']));
    const buf_176 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.bias']));
    const buf_177 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.weight']));
    const buf_178 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.bias']));
    const buf_179 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.weight']));
    const buf_180 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.bias']));
    const buf_181 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight']));
    const buf_182 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.bias']));
    const buf_183 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.weight']));
    const buf_184 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.bias']));
    const buf_185 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight']));
    const buf_186 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.bias']));
    const buf_187 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.weight']));
    const buf_188 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.bias']));
    const buf_189 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.weight']));
    const buf_190 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.bias']));
    const buf_191 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.weight']));
    const buf_192 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.bias']));
    const buf_193 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.weight']));
    const buf_194 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.bias']));
    const buf_195 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.weight']));
    const buf_196 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.bias']));
    const buf_197 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight']));
    const buf_198 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.bias']));
    const buf_199 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.weight']));
    const buf_200 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.bias']));
    const buf_201 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.weight']));
    const buf_202 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.bias']));
    const buf_203 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.weight']));
    const buf_204 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.bias']));
    const buf_205 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.weight']));
    const buf_206 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.bias']));
    const buf_207 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.weight']));
    const buf_208 = createWeightBuf(device, 12288, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.bias']));
    const buf_209 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.weight']));
    const buf_210 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.bias']));
    const output0 = createEmptyBuf(device, 236544);;
    const buf_211 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.final_layer_norm.weight']));
    const buf_212 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['cond_stage_model.transformer.text_model.final_layer_norm.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_386_32_12352_4_4, r_77_77n1, E_77_77, r_77_32_4_8_16_193_3_4, r_77_16_16_77_3n1, r_1848_32_16_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_16_16_192_3_4, r_77_77_4_16_3_4, r_231_4_77, r_231_4_77n1, E_77_77_4_3, r_3_77_4_16_77_4, r_77_16_16_192_3_4n1, r_77_16_48, r_77_16_48n1, E_77_16_16_3, r_77_64_16_192_3_4, r_77_16_16_768_3_4, r_77_16_48, r_77_16_48n1, E_77_16_16_3];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0], [386, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_1], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_3, buf_0, input0, buf_4], [4, 32, 77]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_5, buf_6, buf_1, buf_7], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_8, buf_3, buf_5], [1848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_9, buf_8], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_10, buf_8, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_5, buf_8, buf_9, buf_10, buf_11, buf_12], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_13, buf_5, buf_14, buf_15], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_16, buf_5, buf_17, buf_18], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_19, buf_5, buf_20, buf_21], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_22, buf_13, buf_16, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_5, buf_25, buf_19], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_13, buf_8, buf_5, buf_26, buf_27], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_9, buf_13], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_10, buf_13, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_8, buf_13, buf_9, buf_10, buf_28, buf_29], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_30, buf_8, buf_31, buf_32], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_16, buf_13, buf_30, buf_33, buf_34], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_9, buf_16], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_10, buf_16, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_19, buf_16, buf_9, buf_10, buf_35, buf_36], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_5, buf_19, buf_37, buf_38], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_13, buf_19, buf_39, buf_40], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_8, buf_19, buf_41, buf_42], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_22, buf_5, buf_13, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_19, buf_25, buf_8], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_5, buf_16, buf_19, buf_43, buf_44], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_9, buf_5], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_10, buf_5, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_16, buf_5, buf_9, buf_10, buf_45, buf_46], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_30, buf_16, buf_47, buf_48], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_13, buf_5, buf_30, buf_49, buf_50], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_9, buf_13], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_10, buf_13, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_8, buf_13, buf_9, buf_10, buf_51, buf_52], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_19, buf_8, buf_53, buf_54], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_5, buf_8, buf_55, buf_56], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_16, buf_8, buf_57, buf_58], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_22, buf_19, buf_5, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_8, buf_25, buf_16], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_19, buf_13, buf_8, buf_59, buf_60], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_9, buf_19], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_10, buf_19, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_13, buf_19, buf_9, buf_10, buf_61, buf_62], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_30, buf_13, buf_63, buf_64], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_5, buf_19, buf_30, buf_65, buf_66], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_9, buf_5], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_10, buf_5, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_16, buf_5, buf_9, buf_10, buf_67, buf_68], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_8, buf_16, buf_69, buf_70], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_19, buf_16, buf_71, buf_72], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_13, buf_16, buf_73, buf_74], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_22, buf_8, buf_19, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_16, buf_25, buf_13], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_8, buf_5, buf_16, buf_75, buf_76], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_9, buf_8], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_10, buf_8, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_5, buf_8, buf_9, buf_10, buf_77, buf_78], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_30, buf_5, buf_79, buf_80], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_19, buf_8, buf_30, buf_81, buf_82], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_9, buf_19], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_10, buf_19, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_13, buf_19, buf_9, buf_10, buf_83, buf_84], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_16, buf_13, buf_85, buf_86], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_8, buf_13, buf_87, buf_88], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_5, buf_13, buf_89, buf_90], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_22, buf_16, buf_8, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_13, buf_25, buf_5], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_16, buf_19, buf_13, buf_91, buf_92], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_9, buf_16], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_10, buf_16, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_19, buf_16, buf_9, buf_10, buf_93, buf_94], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_30, buf_19, buf_95, buf_96], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_8, buf_16, buf_30, buf_97, buf_98], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_9, buf_8], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_10, buf_8, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_5, buf_8, buf_9, buf_10, buf_99, buf_100], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_13, buf_5, buf_101, buf_102], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_16, buf_5, buf_103, buf_104], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_19, buf_5, buf_105, buf_106], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_22, buf_13, buf_16, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_5, buf_25, buf_19], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_13, buf_8, buf_5, buf_107, buf_108], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_9, buf_13], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_10, buf_13, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_8, buf_13, buf_9, buf_10, buf_109, buf_110], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_30, buf_8, buf_111, buf_112], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_16, buf_13, buf_30, buf_113, buf_114], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_9, buf_16], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_10, buf_16, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_19, buf_16, buf_9, buf_10, buf_115, buf_116], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_5, buf_19, buf_117, buf_118], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_13, buf_19, buf_119, buf_120], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_8, buf_19, buf_121, buf_122], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_22, buf_5, buf_13, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_19, buf_25, buf_8], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_5, buf_16, buf_19, buf_123, buf_124], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_9, buf_5], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_10, buf_5, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_16, buf_5, buf_9, buf_10, buf_125, buf_126], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_30, buf_16, buf_127, buf_128], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_13, buf_5, buf_30, buf_129, buf_130], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_9, buf_13], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_10, buf_13, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_8, buf_13, buf_9, buf_10, buf_131, buf_132], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_19, buf_8, buf_133, buf_134], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_5, buf_8, buf_135, buf_136], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_16, buf_8, buf_137, buf_138], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_22, buf_19, buf_5, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_8, buf_25, buf_16], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_19, buf_13, buf_8, buf_139, buf_140], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_9, buf_19], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_10, buf_19, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_13, buf_19, buf_9, buf_10, buf_141, buf_142], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_30, buf_13, buf_143, buf_144], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_5, buf_19, buf_30, buf_145, buf_146], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_9, buf_5], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_10, buf_5, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_16, buf_5, buf_9, buf_10, buf_147, buf_148], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_8, buf_16, buf_149, buf_150], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_19, buf_16, buf_151, buf_152], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_13, buf_16, buf_153, buf_154], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_22, buf_8, buf_19, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_16, buf_25, buf_13], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_8, buf_5, buf_16, buf_155, buf_156], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_9, buf_8], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_10, buf_8, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_5, buf_8, buf_9, buf_10, buf_157, buf_158], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_30, buf_5, buf_159, buf_160], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_19, buf_8, buf_30, buf_161, buf_162], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_9, buf_19], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_10, buf_19, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_13, buf_19, buf_9, buf_10, buf_163, buf_164], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_16, buf_13, buf_165, buf_166], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_8, buf_13, buf_167, buf_168], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_5, buf_13, buf_169, buf_170], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_22, buf_16, buf_8, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_13, buf_25, buf_5], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [buf_16, buf_19, buf_13, buf_171, buf_172], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[171], layouts[171], infinityBuf, [buf_9, buf_16], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[172], layouts[172], infinityBuf, [buf_10, buf_16, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[173], layouts[173], infinityBuf, [buf_19, buf_16, buf_9, buf_10, buf_173, buf_174], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[174], layouts[174], infinityBuf, [buf_30, buf_19, buf_175, buf_176], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[175], layouts[175], infinityBuf, [buf_8, buf_16, buf_30, buf_177, buf_178], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[176], layouts[176], infinityBuf, [buf_9, buf_8], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[177], layouts[177], infinityBuf, [buf_10, buf_8, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[178], layouts[178], infinityBuf, [buf_5, buf_8, buf_9, buf_10, buf_179, buf_180], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[179], layouts[179], infinityBuf, [buf_13, buf_5, buf_181, buf_182], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[180], layouts[180], infinityBuf, [buf_16, buf_5, buf_183, buf_184], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[181], layouts[181], infinityBuf, [buf_19, buf_5, buf_185, buf_186], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[182], layouts[182], infinityBuf, [buf_22, buf_13, buf_16, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[183], layouts[183], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[184], layouts[184], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[185], layouts[185], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[186], layouts[186], infinityBuf, [buf_5, buf_25, buf_19], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[187], layouts[187], infinityBuf, [buf_13, buf_8, buf_5, buf_187, buf_188], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[188], layouts[188], infinityBuf, [buf_9, buf_13], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[189], layouts[189], infinityBuf, [buf_10, buf_13, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[190], layouts[190], infinityBuf, [buf_8, buf_13, buf_9, buf_10, buf_189, buf_190], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[191], layouts[191], infinityBuf, [buf_30, buf_8, buf_191, buf_192], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[192], layouts[192], infinityBuf, [buf_16, buf_13, buf_30, buf_193, buf_194], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[193], layouts[193], infinityBuf, [buf_9, buf_16], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[194], layouts[194], infinityBuf, [buf_10, buf_16, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[195], layouts[195], infinityBuf, [buf_19, buf_16, buf_9, buf_10, buf_195, buf_196], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[196], layouts[196], infinityBuf, [buf_5, buf_19, buf_197, buf_198], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[197], layouts[197], infinityBuf, [buf_13, buf_19, buf_199, buf_200], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[198], layouts[198], infinityBuf, [buf_8, buf_19, buf_201, buf_202], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[199], layouts[199], infinityBuf, [buf_22, buf_5, buf_13, buf_2], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[200], layouts[200], infinityBuf, [buf_23, buf_22], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[201], layouts[201], infinityBuf, [buf_24, buf_22, buf_23], [231, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[202], layouts[202], infinityBuf, [buf_25, buf_22, buf_23, buf_24], [77, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[203], layouts[203], infinityBuf, [buf_19, buf_25, buf_8], [77, 3, 1]);
        addComputePass(device, commandEncoder, pipelines[204], layouts[204], infinityBuf, [buf_5, buf_16, buf_19, buf_203, buf_204], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[205], layouts[205], infinityBuf, [buf_9, buf_5], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[206], layouts[206], infinityBuf, [buf_10, buf_5, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[207], layouts[207], infinityBuf, [buf_16, buf_5, buf_9, buf_10, buf_205, buf_206], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[208], layouts[208], infinityBuf, [buf_30, buf_16, buf_207, buf_208], [64, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[209], layouts[209], infinityBuf, [buf_13, buf_5, buf_30, buf_209, buf_210], [16, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[210], layouts[210], infinityBuf, [buf_9, buf_13], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[211], layouts[211], infinityBuf, [buf_10, buf_13, buf_9], [77, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[212], layouts[212], infinityBuf, [output0, buf_13, buf_9, buf_10, buf_211, buf_212], [16, 77, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => {
  if (weight_path instanceof Uint8Array) {
    // If weight_path is already a Uint8Array, use it directly
    return setupNet(device, weight_path);
  } else {
    // Otherwise, fetch and process the data
    return fetch(weight_path)
      .then(response => response.arrayBuffer())
      .then(buffer => setupNet(device, new Uint8Array(buffer)));
  }
};
const getWeight = (safetensor, key) => {
  let uint8Data = getTensorBuffer(safetensor, getTensorMetadata(safetensor)[key], key);
  return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, uint8Data.byteLength / Float32Array.BYTES_PER_ELEMENT);
}
return { load };
})();
export default textModel;
