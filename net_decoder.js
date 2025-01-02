
const decoder = (() => {
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

const E_128_32_4n4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val0 = data1[alu0];
  var alu1 = (alu0+1);
  var val1 = data1[alu1];
  var alu2 = (alu0+2);
  var val2 = data1[alu2];
  var alu3 = (alu0+3);
  var val3 = data1[alu3];
  data0[alu1] = (val1*5.489980785067252f);
  data0[alu2] = (val2*5.489980785067252f);
  data0[alu3] = (val3*5.489980785067252f);
  data0[alu0] = (val0*5.489980785067252f);
}`;

const r_32_32_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 32 */
  var val0 = data2[0];
  var val1 = data2[1];
  var val2 = data2[2];
  var val3 = data2[3];
  var val4 = data2[4];
  var val5 = data2[6];
  var val6 = data2[9];
  var val7 = data2[10];
  var val8 = data2[12];
  var val9 = data2[14];
  var val10 = data2[15];
  var val11 = data3[0];
  var val12 = data3[2];
  var val13 = data2[13];
  var val14 = data2[5];
  var val15 = data2[7];
  var val16 = data2[8];
  var val17 = data2[11];
  var val18 = data3[1];
  var val19 = data3[3];
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val20 = data1[alu0];
  var alu1 = (alu0+1);
  var val21 = data1[alu1];
  var alu2 = (alu0+2);
  var val22 = data1[alu2];
  var alu3 = (alu0+3);
  var val23 = data1[alu3];
  var alu4 = (alu0+4096);
  var val24 = data1[alu4];
  var alu5 = (alu0+4097);
  var val25 = data1[alu5];
  var alu6 = (alu0+4098);
  var val26 = data1[alu6];
  var alu7 = (alu0+4099);
  var val27 = data1[alu7];
  var alu8 = (alu0+8192);
  var val28 = data1[alu8];
  var alu9 = (alu0+8193);
  var val29 = data1[alu9];
  var alu10 = (alu0+8194);
  var val30 = data1[alu10];
  var alu11 = (alu0+8195);
  var val31 = data1[alu11];
  var alu12 = (alu0+12288);
  var val32 = data1[alu12];
  var alu13 = (alu0+12289);
  var val33 = data1[alu13];
  var alu14 = (alu0+12290);
  var val34 = data1[alu14];
  var alu15 = (alu0+12291);
  var val35 = data1[alu15];
  data0[alu1] = (val11+(val21*val0)+(val25*val1)+(val29*val2)+(val33*val3));
  data0[alu5] = (val18+(val21*val4)+(val25*val14)+(val29*val5)+(val33*val15));
  data0[alu9] = (val12+(val21*val16)+(val25*val6)+(val29*val7)+(val33*val17));
  data0[alu13] = (val19+(val21*val8)+(val25*val13)+(val29*val9)+(val33*val10));
  data0[alu2] = (val11+(val22*val0)+(val26*val1)+(val30*val2)+(val34*val3));
  data0[alu6] = (val18+(val22*val4)+(val26*val14)+(val30*val5)+(val34*val15));
  data0[alu10] = (val12+(val22*val16)+(val26*val6)+(val30*val7)+(val34*val17));
  data0[alu14] = (val19+(val22*val8)+(val26*val13)+(val30*val9)+(val34*val10));
  data0[alu3] = (val11+(val23*val0)+(val27*val1)+(val31*val2)+(val35*val3));
  data0[alu7] = (val18+(val23*val4)+(val27*val14)+(val31*val5)+(val35*val15));
  data0[alu11] = (val12+(val23*val16)+(val27*val6)+(val31*val7)+(val35*val17));
  data0[alu15] = (val19+(val23*val8)+(val27*val13)+(val31*val9)+(val35*val10));
  data0[alu0] = (val11+(val24*val1)+(val20*val0)+(val28*val2)+(val32*val3));
  data0[alu4] = (val18+(val24*val14)+(val20*val4)+(val28*val5)+(val32*val15));
  data0[alu8] = (val12+(val24*val6)+(val20*val16)+(val28*val7)+(val32*val17));
  data0[alu12] = (val19+(val24*val13)+(val20*val8)+(val28*val9)+(val32*val10));
}`;

const r_128_8_8_16_4_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<6);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<3))<63);
  var alu6 = (lidx1<15);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 4; ridx0++) {
    var alu7 = ((gidx1*144)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+36)];
    var val10 = data2[(alu7+37)];
    var val11 = data2[(alu7+38)];
    var val12 = data2[(alu7+39)];
    var val13 = data2[(alu7+40)];
    var val14 = data2[(alu7+41)];
    var val15 = data2[(alu7+42)];
    var val16 = data2[(alu7+43)];
    var val17 = data2[(alu7+44)];
    var val18 = data2[(alu7+72)];
    var val19 = data2[(alu7+73)];
    var val20 = data2[(alu7+74)];
    var val21 = data2[(alu7+75)];
    var val22 = data2[(alu7+76)];
    var val23 = data2[(alu7+77)];
    var val24 = data2[(alu7+78)];
    var val25 = data2[(alu7+79)];
    var val26 = data2[(alu7+80)];
    var val27 = data2[(alu7+108)];
    var val28 = data2[(alu7+109)];
    var val29 = data2[(alu7+110)];
    var val30 = data2[(alu7+111)];
    var val31 = data2[(alu7+112)];
    var val32 = data2[(alu7+113)];
    var val33 = data2[(alu7+114)];
    var val34 = data2[(alu7+115)];
    var val35 = data2[(alu7+116)];
    var alu8 = (alu0+alu1+(ridx0<<12)+alu2);
    var val36 = data1[alu8];
    var val37 = select(0.0f, data1[(alu8+-65)], (alu3&alu4));
    var val38 = select(0.0f, data1[(alu8+-64)], alu4);
    var val39 = select(0.0f, data1[(alu8+-63)], alu4);
    var val40 = select(0.0f, data1[(alu8+-62)], alu4);
    var val41 = select(0.0f, data1[(alu8+-61)], alu4);
    var val42 = select(0.0f, data1[(alu8+-60)], (alu6&alu4));
    var val43 = select(0.0f, data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select(0.0f, data1[(alu8+4)], alu6);
    var val48 = select(0.0f, data1[(alu8+63)], (alu5&alu3));
    var val49 = select(0.0f, data1[(alu8+64)], alu5);
    var val50 = select(0.0f, data1[(alu8+65)], alu5);
    var val51 = select(0.0f, data1[(alu8+66)], alu5);
    var val52 = select(0.0f, data1[(alu8+67)], alu5);
    var val53 = select(0.0f, data1[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data3[(alu26+1)];
  var val56 = data3[(alu26+2)];
  var val57 = data3[(alu26+3)];
  var alu27 = (alu0+(gidx1<<14)+alu1+alu2);
  data0[alu27] = (val54+acc0);
  data0[(alu27+1)] = (val54+acc4);
  data0[(alu27+2)] = (val54+acc8);
  data0[(alu27+3)] = (val54+acc12);
  data0[(alu27+4096)] = (val55+acc1);
  data0[(alu27+4097)] = (val55+acc5);
  data0[(alu27+4098)] = (val55+acc9);
  data0[(alu27+4099)] = (val55+acc13);
  data0[(alu27+8192)] = (val56+acc2);
  data0[(alu27+8193)] = (val56+acc6);
  data0[(alu27+8194)] = (val56+acc10);
  data0[(alu27+8195)] = (val56+acc14);
  data0[(alu27+12288)] = (val57+acc3);
  data0[(alu27+12289)] = (val57+acc7);
  data0[(alu27+12290)] = (val57+acc11);
  data0[(alu27+12291)] = (val57+acc15);
}`;

const r_256_32_64_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu0 = ((gidx0<<13)+(lidx0<<8)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val3+val2+val1+val0);
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_256 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (acc0*1.52587890625e-05f);
  }
}`;

const r_4_4_8_16_64_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu0 = ((gidx0<<14)+(gidx1<<19)+(lidx0<<16)+(lidx1<<10)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+256)];
    var val6 = data1[(alu0+257)];
    var val7 = data1[(alu0+258)];
    var val8 = data1[(alu0+259)];
    var val9 = data1[(alu0+512)];
    var val10 = data1[(alu0+513)];
    var val11 = data1[(alu0+514)];
    var val12 = data1[(alu0+515)];
    var val13 = data1[(alu0+768)];
    var val14 = data1[(alu0+769)];
    var val15 = data1[(alu0+770)];
    var val16 = data1[(alu0+771)];
    var alu1 = (val2-val0);
    var alu2 = (val3-val0);
    var alu3 = (val4-val0);
    var alu4 = (val5-val0);
    var alu5 = (val6-val0);
    var alu6 = (val7-val0);
    var alu7 = (val8-val0);
    var alu8 = (val9-val0);
    var alu9 = (val10-val0);
    var alu10 = (val11-val0);
    var alu11 = (val12-val0);
    var alu12 = (val13-val0);
    var alu13 = (val14-val0);
    var alu14 = (val15-val0);
    var alu15 = (val16-val0);
    var alu16 = (val1-val0);
    acc0 = (acc0+(alu1*alu1)+(alu16*alu16)+(alu2*alu2)+(alu3*alu3));
    acc1 = (acc1+(alu4*alu4)+(alu5*alu5)+(alu6*alu6)+(alu7*alu7));
    acc2 = (acc2+(alu8*alu8)+(alu9*alu9)+(alu10*alu10)+(alu11*alu11));
    acc3 = (acc3+(alu12*alu12)+(alu13*alu13)+(alu14*alu14)+(alu15*alu15));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_32_256n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = sqrt((1/((acc0*1.52587890625e-05f)+1e-05f)));
  }
}`;

const E_64_64_8_16_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<15)+(lidx0<<12)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (gidx1>>1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val7*(val3-val6));
  data0[alu2] = ((1/(exp2(((alu6-alu7)*1.4426950408889634f))+1.0f))*(val1+alu7));
  var alu9 = (val0*val7*(val4-val6));
  data0[alu3] = ((1/(exp2(((alu6-alu9)*1.4426950408889634f))+1.0f))*(val1+alu9));
  var alu11 = (val0*val7*(val5-val6));
  data0[alu4] = ((1/(exp2(((alu6-alu11)*1.4426950408889634f))+1.0f))*(val1+alu11));
  var alu13 = (val0*val7*(val2-val6));
  data0[alu1] = ((1/(exp2(((alu6-alu13)*1.4426950408889634f))+1.0f))*(val1+alu13));
}`;

const r_128_8_8_16_512_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<6);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<3))<63);
  var alu6 = (lidx1<15);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu7 = ((gidx1*18432)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+4608)];
    var val10 = data2[(alu7+4609)];
    var val11 = data2[(alu7+4610)];
    var val12 = data2[(alu7+4611)];
    var val13 = data2[(alu7+4612)];
    var val14 = data2[(alu7+4613)];
    var val15 = data2[(alu7+4614)];
    var val16 = data2[(alu7+4615)];
    var val17 = data2[(alu7+4616)];
    var val18 = data2[(alu7+9216)];
    var val19 = data2[(alu7+9217)];
    var val20 = data2[(alu7+9218)];
    var val21 = data2[(alu7+9219)];
    var val22 = data2[(alu7+9220)];
    var val23 = data2[(alu7+9221)];
    var val24 = data2[(alu7+9222)];
    var val25 = data2[(alu7+9223)];
    var val26 = data2[(alu7+9224)];
    var val27 = data2[(alu7+13824)];
    var val28 = data2[(alu7+13825)];
    var val29 = data2[(alu7+13826)];
    var val30 = data2[(alu7+13827)];
    var val31 = data2[(alu7+13828)];
    var val32 = data2[(alu7+13829)];
    var val33 = data2[(alu7+13830)];
    var val34 = data2[(alu7+13831)];
    var val35 = data2[(alu7+13832)];
    var alu8 = (alu0+alu1+(ridx0<<12)+alu2);
    var val36 = data1[alu8];
    var val37 = select(0.0f, data1[(alu8+-65)], (alu3&alu4));
    var val38 = select(0.0f, data1[(alu8+-64)], alu4);
    var val39 = select(0.0f, data1[(alu8+-63)], alu4);
    var val40 = select(0.0f, data1[(alu8+-62)], alu4);
    var val41 = select(0.0f, data1[(alu8+-61)], alu4);
    var val42 = select(0.0f, data1[(alu8+-60)], (alu6&alu4));
    var val43 = select(0.0f, data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select(0.0f, data1[(alu8+4)], alu6);
    var val48 = select(0.0f, data1[(alu8+63)], (alu5&alu3));
    var val49 = select(0.0f, data1[(alu8+64)], alu5);
    var val50 = select(0.0f, data1[(alu8+65)], alu5);
    var val51 = select(0.0f, data1[(alu8+66)], alu5);
    var val52 = select(0.0f, data1[(alu8+67)], alu5);
    var val53 = select(0.0f, data1[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data3[(alu26+1)];
  var val56 = data3[(alu26+2)];
  var val57 = data3[(alu26+3)];
  var alu27 = (alu0+(gidx1<<14)+alu1+alu2);
  data0[alu27] = (val54+acc0);
  data0[(alu27+1)] = (val54+acc4);
  data0[(alu27+2)] = (val54+acc8);
  data0[(alu27+3)] = (val54+acc12);
  data0[(alu27+4096)] = (val55+acc1);
  data0[(alu27+4097)] = (val55+acc5);
  data0[(alu27+4098)] = (val55+acc9);
  data0[(alu27+4099)] = (val55+acc13);
  data0[(alu27+8192)] = (val56+acc2);
  data0[(alu27+8193)] = (val56+acc6);
  data0[(alu27+8194)] = (val56+acc10);
  data0[(alu27+8195)] = (val56+acc14);
  data0[(alu27+12288)] = (val57+acc3);
  data0[(alu27+12289)] = (val57+acc7);
  data0[(alu27+12290)] = (val57+acc11);
  data0[(alu27+12291)] = (val57+acc15);
}`;

const r_128_8_8_16_512_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<6);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<3))<63);
  var alu6 = (lidx1<15);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu7 = ((gidx1*18432)+(ridx0*9));
    var val0 = data3[alu7];
    var val1 = data3[(alu7+1)];
    var val2 = data3[(alu7+2)];
    var val3 = data3[(alu7+3)];
    var val4 = data3[(alu7+4)];
    var val5 = data3[(alu7+5)];
    var val6 = data3[(alu7+6)];
    var val7 = data3[(alu7+7)];
    var val8 = data3[(alu7+8)];
    var val9 = data3[(alu7+4608)];
    var val10 = data3[(alu7+4609)];
    var val11 = data3[(alu7+4610)];
    var val12 = data3[(alu7+4611)];
    var val13 = data3[(alu7+4612)];
    var val14 = data3[(alu7+4613)];
    var val15 = data3[(alu7+4614)];
    var val16 = data3[(alu7+4615)];
    var val17 = data3[(alu7+4616)];
    var val18 = data3[(alu7+9216)];
    var val19 = data3[(alu7+9217)];
    var val20 = data3[(alu7+9218)];
    var val21 = data3[(alu7+9219)];
    var val22 = data3[(alu7+9220)];
    var val23 = data3[(alu7+9221)];
    var val24 = data3[(alu7+9222)];
    var val25 = data3[(alu7+9223)];
    var val26 = data3[(alu7+9224)];
    var val27 = data3[(alu7+13824)];
    var val28 = data3[(alu7+13825)];
    var val29 = data3[(alu7+13826)];
    var val30 = data3[(alu7+13827)];
    var val31 = data3[(alu7+13828)];
    var val32 = data3[(alu7+13829)];
    var val33 = data3[(alu7+13830)];
    var val34 = data3[(alu7+13831)];
    var val35 = data3[(alu7+13832)];
    var alu8 = (alu0+alu1+(ridx0<<12)+alu2);
    var val36 = data2[alu8];
    var val37 = select(0.0f, data2[(alu8+-65)], (alu3&alu4));
    var val38 = select(0.0f, data2[(alu8+-64)], alu4);
    var val39 = select(0.0f, data2[(alu8+-63)], alu4);
    var val40 = select(0.0f, data2[(alu8+-62)], alu4);
    var val41 = select(0.0f, data2[(alu8+-61)], alu4);
    var val42 = select(0.0f, data2[(alu8+-60)], (alu6&alu4));
    var val43 = select(0.0f, data2[(alu8+-1)], alu3);
    var val44 = data2[(alu8+1)];
    var val45 = data2[(alu8+2)];
    var val46 = data2[(alu8+3)];
    var val47 = select(0.0f, data2[(alu8+4)], alu6);
    var val48 = select(0.0f, data2[(alu8+63)], (alu5&alu3));
    var val49 = select(0.0f, data2[(alu8+64)], alu5);
    var val50 = select(0.0f, data2[(alu8+65)], alu5);
    var val51 = select(0.0f, data2[(alu8+66)], alu5);
    var val52 = select(0.0f, data2[(alu8+67)], alu5);
    var val53 = select(0.0f, data2[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu26 = (gidx1<<2);
  var val54 = data4[alu26];
  var val55 = data4[(alu26+1)];
  var val56 = data4[(alu26+2)];
  var val57 = data4[(alu26+3)];
  var alu27 = (alu0+(gidx1<<14)+alu1+alu2);
  var val58 = data1[alu27];
  var alu28 = (alu27+1);
  var val59 = data1[alu28];
  var alu29 = (alu27+2);
  var val60 = data1[alu29];
  var alu30 = (alu27+3);
  var val61 = data1[alu30];
  var alu31 = (alu27+4096);
  var val62 = data1[alu31];
  var alu32 = (alu27+4097);
  var val63 = data1[alu32];
  var alu33 = (alu27+4098);
  var val64 = data1[alu33];
  var alu34 = (alu27+4099);
  var val65 = data1[alu34];
  var alu35 = (alu27+8192);
  var val66 = data1[alu35];
  var alu36 = (alu27+8193);
  var val67 = data1[alu36];
  var alu37 = (alu27+8194);
  var val68 = data1[alu37];
  var alu38 = (alu27+8195);
  var val69 = data1[alu38];
  var alu39 = (alu27+12288);
  var val70 = data1[alu39];
  var alu40 = (alu27+12289);
  var val71 = data1[alu40];
  var alu41 = (alu27+12290);
  var val72 = data1[alu41];
  var alu42 = (alu27+12291);
  var val73 = data1[alu42];
  data0[alu32] = (val63+val55+acc5);
  data0[alu35] = (val66+val56+acc2);
  data0[alu31] = (val62+val55+acc1);
  data0[alu34] = (val65+val55+acc13);
  data0[alu39] = (val70+val57+acc3);
  data0[alu28] = (val59+val54+acc4);
  data0[alu29] = (val60+val54+acc8);
  data0[alu33] = (val64+val55+acc9);
  data0[alu37] = (val68+val56+acc10);
  data0[alu38] = (val69+val56+acc14);
  data0[alu36] = (val67+val56+acc6);
  data0[alu40] = (val71+val57+acc7);
  data0[alu41] = (val72+val57+acc11);
  data0[alu42] = (val73+val57+acc15);
  data0[alu27] = (val58+val54+acc0);
  data0[alu30] = (val61+val54+acc12);
}`;

const E_64_64_8_16_4n1 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<15)+(lidx0<<12)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (gidx1>>1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  data0[alu2] = (val1+(val0*val7*(val3-val6)));
  data0[alu3] = (val1+(val0*val7*(val4-val6)));
  data0[alu4] = (val1+(val0*val7*(val5-val6)));
  data0[alu1] = (val1+(val0*val7*(val2-val6)));
}`;

const r_16_64_8_16_128_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu2 = ((gidx1<<14)+(lidx0<<11)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+512)];
    var val5 = data2[(alu2+513)];
    var val6 = data2[(alu2+514)];
    var val7 = data2[(alu2+515)];
    var val8 = data2[(alu2+1024)];
    var val9 = data2[(alu2+1025)];
    var val10 = data2[(alu2+1026)];
    var val11 = data2[(alu2+1027)];
    var val12 = data2[(alu2+1536)];
    var val13 = data2[(alu2+1537)];
    var val14 = data2[(alu2+1538)];
    var val15 = data2[(alu2+1539)];
    var alu3 = (alu0+alu1+(ridx0<<14));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+4096)];
    var val21 = data1[(alu3+4097)];
    var val22 = data1[(alu3+4098)];
    var val23 = data1[(alu3+4099)];
    var val24 = data1[(alu3+8192)];
    var val25 = data1[(alu3+8193)];
    var val26 = data1[(alu3+8194)];
    var val27 = data1[(alu3+8195)];
    var val28 = data1[(alu3+12288)];
    var val29 = data1[(alu3+12289)];
    var val30 = data1[(alu3+12290)];
    var val31 = data1[(alu3+12291)];
    acc0 = (acc0+(val20*val1)+(val16*val0)+(val24*val2)+(val28*val3));
    acc1 = (acc1+(val20*val5)+(val16*val4)+(val24*val6)+(val28*val7));
    acc2 = (acc2+(val20*val9)+(val16*val8)+(val24*val10)+(val28*val11));
    acc3 = (acc3+(val20*val13)+(val16*val12)+(val24*val14)+(val28*val15));
    acc4 = (acc4+(val17*val0)+(val21*val1)+(val25*val2)+(val29*val3));
    acc5 = (acc5+(val17*val4)+(val21*val5)+(val25*val6)+(val29*val7));
    acc6 = (acc6+(val17*val8)+(val21*val9)+(val25*val10)+(val29*val11));
    acc7 = (acc7+(val17*val12)+(val21*val13)+(val25*val14)+(val29*val15));
    acc8 = (acc8+(val18*val0)+(val22*val1)+(val26*val2)+(val30*val3));
    acc9 = (acc9+(val18*val4)+(val22*val5)+(val26*val6)+(val30*val7));
    acc10 = (acc10+(val18*val8)+(val22*val9)+(val26*val10)+(val30*val11));
    acc11 = (acc11+(val18*val12)+(val22*val13)+(val26*val14)+(val30*val15));
    acc12 = (acc12+(val19*val0)+(val23*val1)+(val27*val2)+(val31*val3));
    acc13 = (acc13+(val19*val4)+(val23*val5)+(val27*val6)+(val31*val7));
    acc14 = (acc14+(val19*val8)+(val23*val9)+(val27*val10)+(val31*val11));
    acc15 = (acc15+(val19*val12)+(val23*val13)+(val27*val14)+(val31*val15));
  }
  var alu21 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data3[(alu21+1)];
  var val34 = data3[(alu21+2)];
  var val35 = data3[(alu21+3)];
  var alu22 = (alu0+(gidx1<<17)+(lidx0<<14)+alu1);
  data0[alu22] = (val32+acc0);
  data0[(alu22+1)] = (val32+acc4);
  data0[(alu22+2)] = (val32+acc8);
  data0[(alu22+3)] = (val32+acc12);
  data0[(alu22+4096)] = (val33+acc1);
  data0[(alu22+4097)] = (val33+acc5);
  data0[(alu22+4098)] = (val33+acc9);
  data0[(alu22+4099)] = (val33+acc13);
  data0[(alu22+8192)] = (val34+acc2);
  data0[(alu22+8193)] = (val34+acc6);
  data0[(alu22+8194)] = (val34+acc10);
  data0[(alu22+8195)] = (val34+acc14);
  data0[(alu22+12288)] = (val35+acc3);
  data0[(alu22+12289)] = (val35+acc7);
  data0[(alu22+12290)] = (val35+acc11);
  data0[(alu22+12291)] = (val35+acc15);
}`;

const r_128_64_8_16_128_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu2 = (ridx0<<14);
    var alu3 = (alu0+alu1+alu2);
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+4096)];
    var val5 = data2[(alu3+4097)];
    var val6 = data2[(alu3+4098)];
    var val7 = data2[(alu3+4099)];
    var val8 = data2[(alu3+8192)];
    var val9 = data2[(alu3+8193)];
    var val10 = data2[(alu3+8194)];
    var val11 = data2[(alu3+8195)];
    var val12 = data2[(alu3+12288)];
    var val13 = data2[(alu3+12289)];
    var val14 = data2[(alu3+12290)];
    var val15 = data2[(alu3+12291)];
    var alu4 = ((gidx1<<5)+(lidx0<<2)+alu2);
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+4096)];
    var val21 = data1[(alu4+4097)];
    var val22 = data1[(alu4+4098)];
    var val23 = data1[(alu4+4099)];
    var val24 = data1[(alu4+8192)];
    var val25 = data1[(alu4+8193)];
    var val26 = data1[(alu4+8194)];
    var val27 = data1[(alu4+8195)];
    var val28 = data1[(alu4+12288)];
    var val29 = data1[(alu4+12289)];
    var val30 = data1[(alu4+12290)];
    var val31 = data1[(alu4+12291)];
    acc0 = (acc0+(val20*val4)+(val16*val0)+(val24*val8)+(val28*val12));
    acc1 = (acc1+(val17*val0)+(val21*val4)+(val25*val8)+(val29*val12));
    acc2 = (acc2+(val18*val0)+(val22*val4)+(val26*val8)+(val30*val12));
    acc3 = (acc3+(val19*val0)+(val23*val4)+(val27*val8)+(val31*val12));
    acc4 = (acc4+(val20*val5)+(val16*val1)+(val24*val9)+(val28*val13));
    acc5 = (acc5+(val17*val1)+(val21*val5)+(val25*val9)+(val29*val13));
    acc6 = (acc6+(val18*val1)+(val22*val5)+(val26*val9)+(val30*val13));
    acc7 = (acc7+(val19*val1)+(val23*val5)+(val27*val9)+(val31*val13));
    acc8 = (acc8+(val20*val6)+(val16*val2)+(val24*val10)+(val28*val14));
    acc9 = (acc9+(val17*val2)+(val21*val6)+(val25*val10)+(val29*val14));
    acc10 = (acc10+(val18*val2)+(val22*val6)+(val26*val10)+(val30*val14));
    acc11 = (acc11+(val19*val2)+(val23*val6)+(val27*val10)+(val31*val14));
    acc12 = (acc12+(val20*val7)+(val16*val3)+(val24*val11)+(val28*val15));
    acc13 = (acc13+(val17*val3)+(val21*val7)+(val25*val11)+(val29*val15));
    acc14 = (acc14+(val18*val3)+(val22*val7)+(val26*val11)+(val30*val15));
    acc15 = (acc15+(val19*val3)+(val23*val7)+(val27*val11)+(val31*val15));
  }
  var alu22 = (alu0+(gidx1<<17)+(lidx0<<14)+alu1);
  data0[alu22] = (acc0*0.04419417306780815f);
  data0[(alu22+1)] = (acc4*0.04419417306780815f);
  data0[(alu22+2)] = (acc8*0.04419417306780815f);
  data0[(alu22+3)] = (acc12*0.04419417306780815f);
  data0[(alu22+4096)] = (acc1*0.04419417306780815f);
  data0[(alu22+4097)] = (acc5*0.04419417306780815f);
  data0[(alu22+4098)] = (acc9*0.04419417306780815f);
  data0[(alu22+4099)] = (acc13*0.04419417306780815f);
  data0[(alu22+8192)] = (acc2*0.04419417306780815f);
  data0[(alu22+8193)] = (acc6*0.04419417306780815f);
  data0[(alu22+8194)] = (acc10*0.04419417306780815f);
  data0[(alu22+8195)] = (acc14*0.04419417306780815f);
  data0[(alu22+12288)] = (acc3*0.04419417306780815f);
  data0[(alu22+12289)] = (acc7*0.04419417306780815f);
  data0[(alu22+12290)] = (acc11*0.04419417306780815f);
  data0[(alu22+12291)] = (acc15*0.04419417306780815f);
}`;

const r_128_32_1024_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 1024; ridx0++) {
    var alu0 = ((gidx0<<17)+(lidx0<<12)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    var alu1 = select(val1,val0,(val1<val0));
    var alu2 = select(val2,alu1,(val2<alu1));
    var alu3 = select(val3,alu2,(val3<alu2));
    acc0 = select(acc0,alu3,(acc0<alu3));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_32_1024_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var alu1 = (alu0+1);
  var alu2 = (alu0+2);
  var alu3 = (alu0+3);
  var val0 = data2[alu1];
  var val1 = data2[alu2];
  var val2 = data2[alu3];
  var val3 = data2[alu0];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1024; ridx0++) {
    var alu4 = ((gidx0<<19)+(lidx0<<14)+(ridx0<<2));
    var val4 = data1[alu4];
    var val5 = data1[(alu4+1)];
    var val6 = data1[(alu4+2)];
    var val7 = data1[(alu4+3)];
    var val8 = data1[(alu4+4096)];
    var val9 = data1[(alu4+4097)];
    var val10 = data1[(alu4+4098)];
    var val11 = data1[(alu4+4099)];
    var val12 = data1[(alu4+8192)];
    var val13 = data1[(alu4+8193)];
    var val14 = data1[(alu4+8194)];
    var val15 = data1[(alu4+8195)];
    var val16 = data1[(alu4+12288)];
    var val17 = data1[(alu4+12289)];
    var val18 = data1[(alu4+12290)];
    var val19 = data1[(alu4+12291)];
    acc0 = (acc0+exp2(((val7-val3)*1.4426950408889634f))+exp2(((val6-val3)*1.4426950408889634f))+exp2(((val5-val3)*1.4426950408889634f))+exp2(((val4-val3)*1.4426950408889634f)));
    acc1 = (acc1+exp2(((val11-val0)*1.4426950408889634f))+exp2(((val10-val0)*1.4426950408889634f))+exp2(((val8-val0)*1.4426950408889634f))+exp2(((val9-val0)*1.4426950408889634f)));
    acc2 = (acc2+exp2(((val15-val1)*1.4426950408889634f))+exp2(((val14-val1)*1.4426950408889634f))+exp2(((val12-val1)*1.4426950408889634f))+exp2(((val13-val1)*1.4426950408889634f)));
    acc3 = (acc3+exp2(((val19-val2)*1.4426950408889634f))+exp2(((val18-val2)*1.4426950408889634f))+exp2(((val16-val2)*1.4426950408889634f))+exp2(((val17-val2)*1.4426950408889634f)));
  }
  data0[alu1] = acc1;
  data0[alu2] = acc2;
  data0[alu3] = acc3;
  data0[alu0] = acc0;
}`;

const E_512_64_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 512 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<15)+(lidx0<<12)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (1/val1);
  data0[alu2] = (exp2(((val3-val0)*1.4426950408889634f))*alu5);
  data0[alu3] = (exp2(((val4-val0)*1.4426950408889634f))*alu5);
  data0[alu4] = (exp2(((val5-val0)*1.4426950408889634f))*alu5);
  data0[alu1] = (exp2(((val2-val0)*1.4426950408889634f))*alu5);
}`;

const r_128_8_8_16_1024_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1024; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0<<18)+(lidx1<<14)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+4096)];
    var val5 = data2[(alu1+4097)];
    var val6 = data2[(alu1+4098)];
    var val7 = data2[(alu1+4099)];
    var val8 = data2[(alu1+8192)];
    var val9 = data2[(alu1+8193)];
    var val10 = data2[(alu1+8194)];
    var val11 = data2[(alu1+8195)];
    var val12 = data2[(alu1+12288)];
    var val13 = data2[(alu1+12289)];
    var val14 = data2[(alu1+12290)];
    var val15 = data2[(alu1+12291)];
    var alu2 = ((gidx1<<17)+(lidx0<<14)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+4096)];
    var val21 = data1[(alu2+4097)];
    var val22 = data1[(alu2+4098)];
    var val23 = data1[(alu2+4099)];
    var val24 = data1[(alu2+8192)];
    var val25 = data1[(alu2+8193)];
    var val26 = data1[(alu2+8194)];
    var val27 = data1[(alu2+8195)];
    var val28 = data1[(alu2+12288)];
    var val29 = data1[(alu2+12289)];
    var val30 = data1[(alu2+12290)];
    var val31 = data1[(alu2+12291)];
    acc0 = (acc0+(val1*val17)+(val0*val16)+(val2*val18)+(val3*val19));
    acc1 = (acc1+(val1*val21)+(val0*val20)+(val2*val22)+(val3*val23));
    acc2 = (acc2+(val1*val25)+(val0*val24)+(val2*val26)+(val3*val27));
    acc3 = (acc3+(val1*val29)+(val0*val28)+(val2*val30)+(val3*val31));
    acc4 = (acc4+(val4*val16)+(val5*val17)+(val6*val18)+(val7*val19));
    acc5 = (acc5+(val4*val20)+(val5*val21)+(val6*val22)+(val7*val23));
    acc6 = (acc6+(val4*val24)+(val5*val25)+(val6*val26)+(val7*val27));
    acc7 = (acc7+(val4*val28)+(val5*val29)+(val6*val30)+(val7*val31));
    acc8 = (acc8+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val19));
    acc9 = (acc9+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc10 = (acc10+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc11 = (acc11+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc12 = (acc12+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val19));
    acc13 = (acc13+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc14 = (acc14+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc15 = (acc15+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
  }
  var alu20 = ((gidx0<<6)+(gidx1<<14)+(lidx0<<11)+(lidx1<<2));
  data0[alu20] = acc0;
  data0[(alu20+1)] = acc4;
  data0[(alu20+2)] = acc8;
  data0[(alu20+3)] = acc12;
  data0[(alu20+512)] = acc1;
  data0[(alu20+513)] = acc5;
  data0[(alu20+514)] = acc9;
  data0[(alu20+515)] = acc13;
  data0[(alu20+1024)] = acc2;
  data0[(alu20+1025)] = acc6;
  data0[(alu20+1026)] = acc10;
  data0[(alu20+1027)] = acc14;
  data0[(alu20+1536)] = acc3;
  data0[(alu20+1537)] = acc7;
  data0[(alu20+1538)] = acc11;
  data0[(alu20+1539)] = acc15;
}`;

const r_16_64_8_16_128_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0<<15)+(lidx1<<11)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+512)];
    var val5 = data2[(alu1+513)];
    var val6 = data2[(alu1+514)];
    var val7 = data2[(alu1+515)];
    var val8 = data2[(alu1+1024)];
    var val9 = data2[(alu1+1025)];
    var val10 = data2[(alu1+1026)];
    var val11 = data2[(alu1+1027)];
    var val12 = data2[(alu1+1536)];
    var val13 = data2[(alu1+1537)];
    var val14 = data2[(alu1+1538)];
    var val15 = data2[(alu1+1539)];
    var alu2 = ((gidx1<<14)+(lidx0<<11)+alu0);
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+512)];
    var val21 = data3[(alu2+513)];
    var val22 = data3[(alu2+514)];
    var val23 = data3[(alu2+515)];
    var val24 = data3[(alu2+1024)];
    var val25 = data3[(alu2+1025)];
    var val26 = data3[(alu2+1026)];
    var val27 = data3[(alu2+1027)];
    var val28 = data3[(alu2+1536)];
    var val29 = data3[(alu2+1537)];
    var val30 = data3[(alu2+1538)];
    var val31 = data3[(alu2+1539)];
    acc0 = (acc0+(val1*val17)+(val0*val16)+(val2*val18)+(val3*val19));
    acc1 = (acc1+(val1*val21)+(val0*val20)+(val2*val22)+(val3*val23));
    acc2 = (acc2+(val1*val25)+(val0*val24)+(val2*val26)+(val3*val27));
    acc3 = (acc3+(val1*val29)+(val0*val28)+(val2*val30)+(val3*val31));
    acc4 = (acc4+(val4*val16)+(val5*val17)+(val6*val18)+(val7*val19));
    acc5 = (acc5+(val4*val20)+(val5*val21)+(val6*val22)+(val7*val23));
    acc6 = (acc6+(val4*val24)+(val5*val25)+(val6*val26)+(val7*val27));
    acc7 = (acc7+(val4*val28)+(val5*val29)+(val6*val30)+(val7*val31));
    acc8 = (acc8+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val19));
    acc9 = (acc9+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc10 = (acc10+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc11 = (acc11+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc12 = (acc12+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val19));
    acc13 = (acc13+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc14 = (acc14+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc15 = (acc15+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
  }
  var alu20 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data4[alu20];
  var val33 = data4[(alu20+1)];
  var val34 = data4[(alu20+2)];
  var val35 = data4[(alu20+3)];
  var alu21 = ((gidx0<<6)+(gidx1<<17)+(lidx0<<14)+(lidx1<<2));
  var val36 = data1[alu21];
  var alu22 = (alu21+1);
  var val37 = data1[alu22];
  var alu23 = (alu21+2);
  var val38 = data1[alu23];
  var alu24 = (alu21+3);
  var val39 = data1[alu24];
  var alu25 = (alu21+4096);
  var val40 = data1[alu25];
  var alu26 = (alu21+4097);
  var val41 = data1[alu26];
  var alu27 = (alu21+4098);
  var val42 = data1[alu27];
  var alu28 = (alu21+4099);
  var val43 = data1[alu28];
  var alu29 = (alu21+8192);
  var val44 = data1[alu29];
  var alu30 = (alu21+8193);
  var val45 = data1[alu30];
  var alu31 = (alu21+8194);
  var val46 = data1[alu31];
  var alu32 = (alu21+8195);
  var val47 = data1[alu32];
  var alu33 = (alu21+12288);
  var val48 = data1[alu33];
  var alu34 = (alu21+12289);
  var val49 = data1[alu34];
  var alu35 = (alu21+12290);
  var val50 = data1[alu35];
  var alu36 = (alu21+12291);
  var val51 = data1[alu36];
  data0[alu26] = (val41+val33+acc5);
  data0[alu27] = (val42+val33+acc9);
  data0[alu29] = (val44+val34+acc2);
  data0[alu33] = (val48+val35+acc3);
  data0[alu34] = (val49+val35+acc7);
  data0[alu35] = (val50+val35+acc11);
  data0[alu25] = (val40+val33+acc1);
  data0[alu31] = (val46+val34+acc10);
  data0[alu30] = (val45+val34+acc6);
  data0[alu23] = (val38+val32+acc8);
  data0[alu32] = (val47+val34+acc14);
  data0[alu36] = (val51+val35+acc15);
  data0[alu22] = (val37+val32+acc4);
  data0[alu24] = (val39+val32+acc12);
  data0[alu28] = (val43+val33+acc13);
  data0[alu21] = (val36+val32+acc0);
}`;

const r_128_16_2_8_16_512_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 16 */
  var gidx2 = i32(gindex.z); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx1+(gidx0<<4));
  var alu1 = (lidx0+((gidx0+((lidx1+1)>>4))>>1));
  var alu2 = ((lidx0+1)>>1);
  var alu3 = (((gidx0+lidx1)<1)!=true);
  var alu4 = (((gidx1+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx1<<3))<127);
  var alu6 = ((alu1+1)>>1);
  var alu7 = (alu0<31);
  var alu8 = ((gidx0<<5)+(lidx1<<1));
  var alu9 = (lidx0+((gidx0+((lidx1+15)>>4)+1)>>1));
  var alu10 = (((alu0+31)&31)<<1);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu11 = ((gidx2*18432)+(ridx0*9));
    var val0 = data2[alu11];
    var val1 = data2[(alu11+1)];
    var val2 = data2[(alu11+2)];
    var val3 = data2[(alu11+3)];
    var val4 = data2[(alu11+4)];
    var val5 = data2[(alu11+5)];
    var val6 = data2[(alu11+6)];
    var val7 = data2[(alu11+7)];
    var val8 = data2[(alu11+8)];
    var val9 = data2[(alu11+4608)];
    var val10 = data2[(alu11+4609)];
    var val11 = data2[(alu11+4610)];
    var val12 = data2[(alu11+4611)];
    var val13 = data2[(alu11+4612)];
    var val14 = data2[(alu11+4613)];
    var val15 = data2[(alu11+4614)];
    var val16 = data2[(alu11+4615)];
    var val17 = data2[(alu11+4616)];
    var val18 = data2[(alu11+9216)];
    var val19 = data2[(alu11+9217)];
    var val20 = data2[(alu11+9218)];
    var val21 = data2[(alu11+9219)];
    var val22 = data2[(alu11+9220)];
    var val23 = data2[(alu11+9221)];
    var val24 = data2[(alu11+9222)];
    var val25 = data2[(alu11+9223)];
    var val26 = data2[(alu11+9224)];
    var val27 = data2[(alu11+13824)];
    var val28 = data2[(alu11+13825)];
    var val29 = data2[(alu11+13826)];
    var val30 = data2[(alu11+13827)];
    var val31 = data2[(alu11+13828)];
    var val32 = data2[(alu11+13829)];
    var val33 = data2[(alu11+13830)];
    var val34 = data2[(alu11+13831)];
    var val35 = data2[(alu11+13832)];
    var alu12 = ((gidx1<<2)+(ridx0<<6));
    var alu13 = (alu12+(alu9>>1));
    var alu14 = ((gidx1<<8)+(ridx0<<12));
    var alu15 = (alu14+((lidx0>>1)<<6)+alu8);
    var val36 = data1[alu15];
    var val37 = data1[(alu15+1)];
    var alu16 = (alu14+(alu2<<6)+alu8);
    var val38 = select(0.0f, data1[(alu16+-64)], alu4);
    var val39 = select(0.0f, data1[(alu16+-63)], alu4);
    var val40 = select(0.0f, data1[(alu14+(alu6<<6)+alu8+-62)], (alu7&alu4));
    var val41 = select(0.0f, data1[(alu10+(((alu13+32767)&32767)<<6)+1)], (alu3&alu4));
    var val42 = select(0.0f, data1[(alu10+(((alu12+((alu9+1)>>1)+32767)&32767)<<6)+1)], alu3);
    var val43 = select(0.0f, data1[(alu8+(((alu12+(alu1>>1))&32767)<<6)+2)], alu7);
    var val44 = select(0.0f, data1[(alu10+((alu13&32767)<<6)+1)], (alu5&alu3));
    var alu17 = (alu8+(((alu12+alu2)&32767)<<6));
    var val45 = select(0.0f, data1[alu17], alu5);
    var val46 = select(0.0f, data1[(alu17+1)], alu5);
    var val47 = select(0.0f, data1[(alu8+(((alu12+alu6)&32767)<<6)+2)], (alu5&alu7));
    var alu18 = (val37*val4);
    var alu19 = (val37*val5);
    var alu20 = (val37*val13);
    var alu21 = (val37*val14);
    var alu22 = (val37*val22);
    var alu23 = (val37*val23);
    var alu24 = (val37*val31);
    var alu25 = (val37*val32);
    var alu26 = (val38*val1);
    var alu27 = (val38*val10);
    var alu28 = (val38*val19);
    var alu29 = (val38*val28);
    var alu30 = (val39*val1);
    var alu31 = (val39*val2);
    var alu32 = (val39*val10);
    var alu33 = (val39*val11);
    var alu34 = (val39*val19);
    var alu35 = (val39*val20);
    var alu36 = (val39*val28);
    var alu37 = (val39*val29);
    var alu38 = (val46*val7);
    var alu39 = (val46*val8);
    var alu40 = (val46*val16);
    var alu41 = (val46*val17);
    var alu42 = (val46*val25);
    var alu43 = (val46*val26);
    var alu44 = (val46*val34);
    var alu45 = (val46*val35);
    var alu46 = (val36*val4);
    var alu47 = (val36*val13);
    var alu48 = (val36*val22);
    var alu49 = (val36*val31);
    var alu50 = ((val38*val0)+(val36*val3)+(val45*val6));
    var alu51 = (val45*val7);
    var alu52 = ((val38*val9)+(val36*val12)+(val45*val15));
    var alu53 = (val45*val16);
    var alu54 = ((val38*val18)+(val36*val21)+(val45*val24));
    var alu55 = (val45*val25);
    var alu56 = ((val38*val27)+(val36*val30)+(val45*val33));
    var alu57 = (val45*val34);
    acc0 = (acc0+(val41*val0)+(val42*val3)+(val44*val6)+alu26+alu46+alu51+(val38*val2)+(val36*val5)+(val45*val8));
    acc1 = (acc1+(val41*val9)+(val42*val12)+(val44*val15)+alu27+alu47+alu53+(val38*val11)+(val36*val14)+(val45*val17));
    acc2 = (acc2+(val41*val18)+(val42*val21)+(val44*val24)+alu28+alu48+alu55+(val38*val20)+(val36*val23)+(val45*val26));
    acc3 = (acc3+(val41*val27)+(val42*val30)+(val44*val33)+alu29+alu49+alu57+(val38*val29)+(val36*val32)+(val45*val35));
    acc4 = (acc4+alu50+alu26+alu46+alu51+alu31+alu19+alu39);
    acc5 = (acc5+alu52+alu27+alu47+alu53+alu33+alu21+alu41);
    acc6 = (acc6+alu54+alu28+alu48+alu55+alu35+alu23+alu43);
    acc7 = (acc7+alu56+alu29+alu49+alu57+alu37+alu25+alu45);
    acc8 = (acc8+alu50+alu30+alu18+alu38+alu31+alu19+alu39);
    acc9 = (acc9+alu52+alu32+alu20+alu40+alu33+alu21+alu41);
    acc10 = (acc10+alu54+alu34+alu22+alu42+alu35+alu23+alu43);
    acc11 = (acc11+alu56+alu36+alu24+alu44+alu37+alu25+alu45);
    acc12 = (acc12+(val37*val3)+(val39*val0)+(val46*val6)+alu30+alu18+alu38+(val40*val2)+(val43*val5)+(val47*val8));
    acc13 = (acc13+(val37*val12)+(val39*val9)+(val46*val15)+alu32+alu20+alu40+(val40*val11)+(val43*val14)+(val47*val17));
    acc14 = (acc14+(val37*val21)+(val39*val18)+(val46*val24)+alu34+alu22+alu42+(val40*val20)+(val43*val23)+(val47*val26));
    acc15 = (acc15+(val37*val30)+(val39*val27)+(val46*val33)+alu36+alu24+alu44+(val40*val29)+(val43*val32)+(val47*val35));
  }
  var alu75 = (gidx2<<2);
  var val48 = data3[alu75];
  var val49 = data3[(alu75+1)];
  var val50 = data3[(alu75+2)];
  var val51 = data3[(alu75+3)];
  var alu76 = ((gidx1<<10)+(gidx2<<16)+(gidx0<<6)+(lidx0<<7)+(lidx1<<2));
  data0[alu76] = (val48+acc0);
  data0[(alu76+1)] = (val48+acc4);
  data0[(alu76+2)] = (val48+acc8);
  data0[(alu76+3)] = (val48+acc12);
  data0[(alu76+16384)] = (val49+acc1);
  data0[(alu76+16385)] = (val49+acc5);
  data0[(alu76+16386)] = (val49+acc9);
  data0[(alu76+16387)] = (val49+acc13);
  data0[(alu76+32768)] = (val50+acc2);
  data0[(alu76+32769)] = (val50+acc6);
  data0[(alu76+32770)] = (val50+acc10);
  data0[(alu76+32771)] = (val50+acc14);
  data0[(alu76+49152)] = (val51+acc3);
  data0[(alu76+49153)] = (val51+acc7);
  data0[(alu76+49154)] = (val51+acc11);
  data0[(alu76+49155)] = (val51+acc15);
}`;

const r_256_32_256_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu0 = ((gidx0<<15)+(lidx0<<10)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val3+val2+val1+val0);
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_256n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (acc0*3.814697265625e-06f);
  }
}`;

const r_4_4_8_16_256_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu0 = ((gidx0<<16)+(gidx1<<21)+(lidx0<<18)+(lidx1<<12)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+1024)];
    var val6 = data1[(alu0+1025)];
    var val7 = data1[(alu0+1026)];
    var val8 = data1[(alu0+1027)];
    var val9 = data1[(alu0+2048)];
    var val10 = data1[(alu0+2049)];
    var val11 = data1[(alu0+2050)];
    var val12 = data1[(alu0+2051)];
    var val13 = data1[(alu0+3072)];
    var val14 = data1[(alu0+3073)];
    var val15 = data1[(alu0+3074)];
    var val16 = data1[(alu0+3075)];
    var alu1 = (val2-val0);
    var alu2 = (val3-val0);
    var alu3 = (val4-val0);
    var alu4 = (val5-val0);
    var alu5 = (val6-val0);
    var alu6 = (val7-val0);
    var alu7 = (val8-val0);
    var alu8 = (val9-val0);
    var alu9 = (val10-val0);
    var alu10 = (val11-val0);
    var alu11 = (val12-val0);
    var alu12 = (val13-val0);
    var alu13 = (val14-val0);
    var alu14 = (val15-val0);
    var alu15 = (val16-val0);
    var alu16 = (val1-val0);
    acc0 = (acc0+(alu1*alu1)+(alu16*alu16)+(alu2*alu2)+(alu3*alu3));
    acc1 = (acc1+(alu4*alu4)+(alu5*alu5)+(alu6*alu6)+(alu7*alu7));
    acc2 = (acc2+(alu8*alu8)+(alu9*alu9)+(alu10*alu10)+(alu11*alu11));
    acc3 = (acc3+(alu12*alu12)+(alu13*alu13)+(alu14*alu14)+(alu15*alu15));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_32_256n3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = sqrt((1/((acc0*3.814697265625e-06f)+1e-05f)));
  }
}`;

const E_64_256_8_16_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<17)+(lidx0<<14)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (gidx1>>1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val7*(val3-val6));
  data0[alu2] = ((1/(exp2(((alu6-alu7)*1.4426950408889634f))+1.0f))*(val1+alu7));
  var alu9 = (val0*val7*(val4-val6));
  data0[alu3] = ((1/(exp2(((alu6-alu9)*1.4426950408889634f))+1.0f))*(val1+alu9));
  var alu11 = (val0*val7*(val5-val6));
  data0[alu4] = ((1/(exp2(((alu6-alu11)*1.4426950408889634f))+1.0f))*(val1+alu11));
  var alu13 = (val0*val7*(val2-val6));
  data0[alu1] = ((1/(exp2(((alu6-alu13)*1.4426950408889634f))+1.0f))*(val1+alu13));
}`;

const r_128_16_2_8_16_512_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 16 */
  var gidx2 = i32(gindex.z); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<10);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<7);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<127);
  var alu7 = ((lidx1+(gidx0<<4))<31);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu8 = ((gidx2*18432)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+4608)];
    var val10 = data2[(alu8+4609)];
    var val11 = data2[(alu8+4610)];
    var val12 = data2[(alu8+4611)];
    var val13 = data2[(alu8+4612)];
    var val14 = data2[(alu8+4613)];
    var val15 = data2[(alu8+4614)];
    var val16 = data2[(alu8+4615)];
    var val17 = data2[(alu8+4616)];
    var val18 = data2[(alu8+9216)];
    var val19 = data2[(alu8+9217)];
    var val20 = data2[(alu8+9218)];
    var val21 = data2[(alu8+9219)];
    var val22 = data2[(alu8+9220)];
    var val23 = data2[(alu8+9221)];
    var val24 = data2[(alu8+9222)];
    var val25 = data2[(alu8+9223)];
    var val26 = data2[(alu8+9224)];
    var val27 = data2[(alu8+13824)];
    var val28 = data2[(alu8+13825)];
    var val29 = data2[(alu8+13826)];
    var val30 = data2[(alu8+13827)];
    var val31 = data2[(alu8+13828)];
    var val32 = data2[(alu8+13829)];
    var val33 = data2[(alu8+13830)];
    var val34 = data2[(alu8+13831)];
    var val35 = data2[(alu8+13832)];
    var alu9 = (alu0+alu2+(ridx0<<14)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-129)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-128)], alu5);
    var val39 = select(0.0f, data1[(alu9+-127)], alu5);
    var val40 = select(0.0f, data1[(alu9+-126)], alu5);
    var val41 = select(0.0f, data1[(alu9+-125)], alu5);
    var val42 = select(0.0f, data1[(alu9+-124)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+127)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+128)], alu6);
    var val50 = select(0.0f, data1[(alu9+129)], alu6);
    var val51 = select(0.0f, data1[(alu9+130)], alu6);
    var val52 = select(0.0f, data1[(alu9+131)], alu6);
    var val53 = select(0.0f, data1[(alu9+132)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data3[alu27];
  var val55 = data3[(alu27+1)];
  var val56 = data3[(alu27+2)];
  var val57 = data3[(alu27+3)];
  var alu28 = (alu0+(gidx2<<16)+alu1+alu2+alu3);
  data0[alu28] = (val54+acc0);
  data0[(alu28+1)] = (val54+acc4);
  data0[(alu28+2)] = (val54+acc8);
  data0[(alu28+3)] = (val54+acc12);
  data0[(alu28+16384)] = (val55+acc1);
  data0[(alu28+16385)] = (val55+acc5);
  data0[(alu28+16386)] = (val55+acc9);
  data0[(alu28+16387)] = (val55+acc13);
  data0[(alu28+32768)] = (val56+acc2);
  data0[(alu28+32769)] = (val56+acc6);
  data0[(alu28+32770)] = (val56+acc10);
  data0[(alu28+32771)] = (val56+acc14);
  data0[(alu28+49152)] = (val57+acc3);
  data0[(alu28+49153)] = (val57+acc7);
  data0[(alu28+49154)] = (val57+acc11);
  data0[(alu28+49155)] = (val57+acc15);
}`;

const r_128_16_2_8_16_512_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 16 */
  var gidx2 = i32(gindex.z); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<10);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<7);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<127);
  var alu7 = ((lidx1+(gidx0<<4))<31);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu8 = ((gidx2*18432)+(ridx0*9));
    var val0 = data3[alu8];
    var val1 = data3[(alu8+1)];
    var val2 = data3[(alu8+2)];
    var val3 = data3[(alu8+3)];
    var val4 = data3[(alu8+4)];
    var val5 = data3[(alu8+5)];
    var val6 = data3[(alu8+6)];
    var val7 = data3[(alu8+7)];
    var val8 = data3[(alu8+8)];
    var val9 = data3[(alu8+4608)];
    var val10 = data3[(alu8+4609)];
    var val11 = data3[(alu8+4610)];
    var val12 = data3[(alu8+4611)];
    var val13 = data3[(alu8+4612)];
    var val14 = data3[(alu8+4613)];
    var val15 = data3[(alu8+4614)];
    var val16 = data3[(alu8+4615)];
    var val17 = data3[(alu8+4616)];
    var val18 = data3[(alu8+9216)];
    var val19 = data3[(alu8+9217)];
    var val20 = data3[(alu8+9218)];
    var val21 = data3[(alu8+9219)];
    var val22 = data3[(alu8+9220)];
    var val23 = data3[(alu8+9221)];
    var val24 = data3[(alu8+9222)];
    var val25 = data3[(alu8+9223)];
    var val26 = data3[(alu8+9224)];
    var val27 = data3[(alu8+13824)];
    var val28 = data3[(alu8+13825)];
    var val29 = data3[(alu8+13826)];
    var val30 = data3[(alu8+13827)];
    var val31 = data3[(alu8+13828)];
    var val32 = data3[(alu8+13829)];
    var val33 = data3[(alu8+13830)];
    var val34 = data3[(alu8+13831)];
    var val35 = data3[(alu8+13832)];
    var alu9 = (alu0+alu2+(ridx0<<14)+alu1+alu3);
    var val36 = data2[alu9];
    var val37 = select(0.0f, data2[(alu9+-129)], (alu4&alu5));
    var val38 = select(0.0f, data2[(alu9+-128)], alu5);
    var val39 = select(0.0f, data2[(alu9+-127)], alu5);
    var val40 = select(0.0f, data2[(alu9+-126)], alu5);
    var val41 = select(0.0f, data2[(alu9+-125)], alu5);
    var val42 = select(0.0f, data2[(alu9+-124)], (alu7&alu5));
    var val43 = select(0.0f, data2[(alu9+-1)], alu4);
    var val44 = data2[(alu9+1)];
    var val45 = data2[(alu9+2)];
    var val46 = data2[(alu9+3)];
    var val47 = select(0.0f, data2[(alu9+4)], alu7);
    var val48 = select(0.0f, data2[(alu9+127)], (alu6&alu4));
    var val49 = select(0.0f, data2[(alu9+128)], alu6);
    var val50 = select(0.0f, data2[(alu9+129)], alu6);
    var val51 = select(0.0f, data2[(alu9+130)], alu6);
    var val52 = select(0.0f, data2[(alu9+131)], alu6);
    var val53 = select(0.0f, data2[(alu9+132)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data4[alu27];
  var val55 = data4[(alu27+1)];
  var val56 = data4[(alu27+2)];
  var val57 = data4[(alu27+3)];
  var alu28 = (alu0+(gidx2<<16)+alu1+alu2+alu3);
  var val58 = data1[alu28];
  var alu29 = (alu28+1);
  var val59 = data1[alu29];
  var alu30 = (alu28+2);
  var val60 = data1[alu30];
  var alu31 = (alu28+3);
  var val61 = data1[alu31];
  var alu32 = (alu28+16384);
  var val62 = data1[alu32];
  var alu33 = (alu28+16385);
  var val63 = data1[alu33];
  var alu34 = (alu28+16386);
  var val64 = data1[alu34];
  var alu35 = (alu28+16387);
  var val65 = data1[alu35];
  var alu36 = (alu28+32768);
  var val66 = data1[alu36];
  var alu37 = (alu28+32769);
  var val67 = data1[alu37];
  var alu38 = (alu28+32770);
  var val68 = data1[alu38];
  var alu39 = (alu28+32771);
  var val69 = data1[alu39];
  var alu40 = (alu28+49152);
  var val70 = data1[alu40];
  var alu41 = (alu28+49153);
  var val71 = data1[alu41];
  var alu42 = (alu28+49154);
  var val72 = data1[alu42];
  var alu43 = (alu28+49155);
  var val73 = data1[alu43];
  data0[alu33] = (val63+val55+acc5);
  data0[alu36] = (val66+val56+acc2);
  data0[alu32] = (val62+val55+acc1);
  data0[alu35] = (val65+val55+acc13);
  data0[alu40] = (val70+val57+acc3);
  data0[alu29] = (val59+val54+acc4);
  data0[alu30] = (val60+val54+acc8);
  data0[alu34] = (val64+val55+acc9);
  data0[alu38] = (val68+val56+acc10);
  data0[alu39] = (val69+val56+acc14);
  data0[alu37] = (val67+val56+acc6);
  data0[alu41] = (val71+val57+acc7);
  data0[alu42] = (val72+val57+acc11);
  data0[alu43] = (val73+val57+acc15);
  data0[alu28] = (val58+val54+acc0);
  data0[alu31] = (val61+val54+acc12);
}`;

const r_128_32_4_8_16_512_4_4_3_3 = `enable f16;
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
  var gidx2 = i32(gindex.z); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+((gidx0+((lidx1+1)>>4))>>2));
  var alu1 = ((lidx0+1)>>1);
  var alu2 = (((gidx0+lidx1)<1)!=true);
  var alu3 = (((gidx1+lidx0)<1)!=true);
  var alu4 = ((lidx0+(gidx1<<3))<255);
  var alu5 = ((alu0+1)>>1);
  var alu6 = ((lidx1+(gidx0<<4))<63);
  var alu7 = ((gidx0<<5)+(lidx1<<1));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu8 = ((gidx2*18432)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+4608)];
    var val10 = data2[(alu8+4609)];
    var val11 = data2[(alu8+4610)];
    var val12 = data2[(alu8+4611)];
    var val13 = data2[(alu8+4612)];
    var val14 = data2[(alu8+4613)];
    var val15 = data2[(alu8+4614)];
    var val16 = data2[(alu8+4615)];
    var val17 = data2[(alu8+4616)];
    var val18 = data2[(alu8+9216)];
    var val19 = data2[(alu8+9217)];
    var val20 = data2[(alu8+9218)];
    var val21 = data2[(alu8+9219)];
    var val22 = data2[(alu8+9220)];
    var val23 = data2[(alu8+9221)];
    var val24 = data2[(alu8+9222)];
    var val25 = data2[(alu8+9223)];
    var val26 = data2[(alu8+9224)];
    var val27 = data2[(alu8+13824)];
    var val28 = data2[(alu8+13825)];
    var val29 = data2[(alu8+13826)];
    var val30 = data2[(alu8+13827)];
    var val31 = data2[(alu8+13828)];
    var val32 = data2[(alu8+13829)];
    var val33 = data2[(alu8+13830)];
    var val34 = data2[(alu8+13831)];
    var val35 = data2[(alu8+13832)];
    var alu9 = ((gidx1<<2)+(ridx0<<7));
    var alu10 = ((gidx1<<9)+(ridx0<<14));
    var alu11 = (alu10+((lidx0>>1)<<7)+alu7);
    var val36 = data1[alu11];
    var val37 = select(0.0f, data1[(alu11+-1)], alu2);
    var val38 = data1[(alu11+1)];
    var alu12 = (alu10+(alu1<<7)+alu7);
    var val39 = select(0.0f, data1[(alu12+-129)], (alu2&alu3));
    var val40 = select(0.0f, data1[(alu12+-128)], alu3);
    var val41 = select(0.0f, data1[(alu12+-127)], alu3);
    var val42 = select(0.0f, data1[(alu10+(alu5<<7)+alu7+-126)], (alu6&alu3));
    var val43 = select(0.0f, data1[(alu7+(((alu9+(alu0>>1))&65535)<<7)+2)], alu6);
    var alu13 = (alu7+(((alu9+alu1)&65535)<<7));
    var val44 = select(0.0f, data1[alu13], alu4);
    var val45 = select(0.0f, data1[(alu13+-1)], (alu4&alu2));
    var val46 = select(0.0f, data1[(alu13+1)], alu4);
    var val47 = select(0.0f, data1[(alu7+(((alu9+alu5)&65535)<<7)+2)], (alu4&alu6));
    var alu14 = (val38*val4);
    var alu15 = (val38*val5);
    var alu16 = (val38*val13);
    var alu17 = (val38*val14);
    var alu18 = (val38*val22);
    var alu19 = (val38*val23);
    var alu20 = (val38*val31);
    var alu21 = (val38*val32);
    var alu22 = (val40*val1);
    var alu23 = (val40*val10);
    var alu24 = (val40*val19);
    var alu25 = (val40*val28);
    var alu26 = (val41*val1);
    var alu27 = (val41*val2);
    var alu28 = (val41*val10);
    var alu29 = (val41*val11);
    var alu30 = (val41*val19);
    var alu31 = (val41*val20);
    var alu32 = (val41*val28);
    var alu33 = (val41*val29);
    var alu34 = (val46*val7);
    var alu35 = (val46*val8);
    var alu36 = (val46*val16);
    var alu37 = (val46*val17);
    var alu38 = (val46*val25);
    var alu39 = (val46*val26);
    var alu40 = (val46*val34);
    var alu41 = (val46*val35);
    var alu42 = (val36*val4);
    var alu43 = (val36*val13);
    var alu44 = (val36*val22);
    var alu45 = (val36*val31);
    var alu46 = ((val40*val0)+(val36*val3)+(val44*val6));
    var alu47 = (val44*val7);
    var alu48 = ((val40*val9)+(val36*val12)+(val44*val15));
    var alu49 = (val44*val16);
    var alu50 = ((val40*val18)+(val36*val21)+(val44*val24));
    var alu51 = (val44*val25);
    var alu52 = ((val40*val27)+(val36*val30)+(val44*val33));
    var alu53 = (val44*val34);
    acc0 = (acc0+(val37*val3)+(val39*val0)+(val45*val6)+alu22+alu42+alu47+(val40*val2)+(val36*val5)+(val44*val8));
    acc1 = (acc1+(val37*val12)+(val39*val9)+(val45*val15)+alu23+alu43+alu49+(val40*val11)+(val36*val14)+(val44*val17));
    acc2 = (acc2+(val37*val21)+(val39*val18)+(val45*val24)+alu24+alu44+alu51+(val40*val20)+(val36*val23)+(val44*val26));
    acc3 = (acc3+(val37*val30)+(val39*val27)+(val45*val33)+alu25+alu45+alu53+(val40*val29)+(val36*val32)+(val44*val35));
    acc4 = (acc4+alu46+alu22+alu42+alu47+alu27+alu15+alu35);
    acc5 = (acc5+alu48+alu23+alu43+alu49+alu29+alu17+alu37);
    acc6 = (acc6+alu50+alu24+alu44+alu51+alu31+alu19+alu39);
    acc7 = (acc7+alu52+alu25+alu45+alu53+alu33+alu21+alu41);
    acc8 = (acc8+alu46+alu26+alu14+alu34+alu27+alu15+alu35);
    acc9 = (acc9+alu48+alu28+alu16+alu36+alu29+alu17+alu37);
    acc10 = (acc10+alu50+alu30+alu18+alu38+alu31+alu19+alu39);
    acc11 = (acc11+alu52+alu32+alu20+alu40+alu33+alu21+alu41);
    acc12 = (acc12+(val38*val3)+(val41*val0)+(val46*val6)+alu26+alu14+alu34+(val42*val2)+(val43*val5)+(val47*val8));
    acc13 = (acc13+(val38*val12)+(val41*val9)+(val46*val15)+alu28+alu16+alu36+(val42*val11)+(val43*val14)+(val47*val17));
    acc14 = (acc14+(val38*val21)+(val41*val18)+(val46*val24)+alu30+alu18+alu38+(val42*val20)+(val43*val23)+(val47*val26));
    acc15 = (acc15+(val38*val30)+(val41*val27)+(val46*val33)+alu32+alu20+alu40+(val42*val29)+(val43*val32)+(val47*val35));
  }
  var alu71 = (gidx2<<2);
  var val48 = data3[alu71];
  var val49 = data3[(alu71+1)];
  var val50 = data3[(alu71+2)];
  var val51 = data3[(alu71+3)];
  var alu72 = ((gidx1<<11)+(gidx2<<18)+(gidx0<<6)+(lidx0<<8)+(lidx1<<2));
  data0[alu72] = (val48+acc0);
  data0[(alu72+1)] = (val48+acc4);
  data0[(alu72+2)] = (val48+acc8);
  data0[(alu72+3)] = (val48+acc12);
  data0[(alu72+65536)] = (val49+acc1);
  data0[(alu72+65537)] = (val49+acc5);
  data0[(alu72+65538)] = (val49+acc9);
  data0[(alu72+65539)] = (val49+acc13);
  data0[(alu72+131072)] = (val50+acc2);
  data0[(alu72+131073)] = (val50+acc6);
  data0[(alu72+131074)] = (val50+acc10);
  data0[(alu72+131075)] = (val50+acc14);
  data0[(alu72+196608)] = (val51+acc3);
  data0[(alu72+196609)] = (val51+acc7);
  data0[(alu72+196610)] = (val51+acc11);
  data0[(alu72+196611)] = (val51+acc15);
}`;

const r_256_32_1024_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1024; ridx0++) {
    var alu0 = ((gidx0<<17)+(lidx0<<12)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val3+val2+val1+val0);
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_256n4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (acc0*9.5367431640625e-07f);
  }
}`;

const r_4_4_8_16_1024_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1024; ridx0++) {
    var alu0 = ((gidx0<<18)+(gidx1<<23)+(lidx0<<20)+(lidx1<<14)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+4096)];
    var val6 = data1[(alu0+4097)];
    var val7 = data1[(alu0+4098)];
    var val8 = data1[(alu0+4099)];
    var val9 = data1[(alu0+8192)];
    var val10 = data1[(alu0+8193)];
    var val11 = data1[(alu0+8194)];
    var val12 = data1[(alu0+8195)];
    var val13 = data1[(alu0+12288)];
    var val14 = data1[(alu0+12289)];
    var val15 = data1[(alu0+12290)];
    var val16 = data1[(alu0+12291)];
    var alu1 = (val2-val0);
    var alu2 = (val3-val0);
    var alu3 = (val4-val0);
    var alu4 = (val5-val0);
    var alu5 = (val6-val0);
    var alu6 = (val7-val0);
    var alu7 = (val8-val0);
    var alu8 = (val9-val0);
    var alu9 = (val10-val0);
    var alu10 = (val11-val0);
    var alu11 = (val12-val0);
    var alu12 = (val13-val0);
    var alu13 = (val14-val0);
    var alu14 = (val15-val0);
    var alu15 = (val16-val0);
    var alu16 = (val1-val0);
    acc0 = (acc0+(alu1*alu1)+(alu16*alu16)+(alu2*alu2)+(alu3*alu3));
    acc1 = (acc1+(alu4*alu4)+(alu5*alu5)+(alu6*alu6)+(alu7*alu7));
    acc2 = (acc2+(alu8*alu8)+(alu9*alu9)+(alu10*alu10)+(alu11*alu11));
    acc3 = (acc3+(alu12*alu12)+(alu13*alu13)+(alu14*alu14)+(alu15*alu15));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_32_256n5 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = sqrt((1/((acc0*9.5367431640625e-07f)+1e-05f)));
  }
}`;

const E_64_1024_8_16_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<19)+(lidx0<<16)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (gidx1>>1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val7*(val3-val6));
  data0[alu2] = ((1/(exp2(((alu6-alu7)*1.4426950408889634f))+1.0f))*(val1+alu7));
  var alu9 = (val0*val7*(val4-val6));
  data0[alu3] = ((1/(exp2(((alu6-alu9)*1.4426950408889634f))+1.0f))*(val1+alu9));
  var alu11 = (val0*val7*(val5-val6));
  data0[alu4] = ((1/(exp2(((alu6-alu11)*1.4426950408889634f))+1.0f))*(val1+alu11));
  var alu13 = (val0*val7*(val2-val6));
  data0[alu1] = ((1/(exp2(((alu6-alu13)*1.4426950408889634f))+1.0f))*(val1+alu13));
}`;

const r_64_32_4_8_16_512_4_4_3_3 = `enable f16;
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
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<11);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<8);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<255);
  var alu7 = ((lidx1+(gidx0<<4))<63);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu8 = ((gidx2*18432)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+4608)];
    var val10 = data2[(alu8+4609)];
    var val11 = data2[(alu8+4610)];
    var val12 = data2[(alu8+4611)];
    var val13 = data2[(alu8+4612)];
    var val14 = data2[(alu8+4613)];
    var val15 = data2[(alu8+4614)];
    var val16 = data2[(alu8+4615)];
    var val17 = data2[(alu8+4616)];
    var val18 = data2[(alu8+9216)];
    var val19 = data2[(alu8+9217)];
    var val20 = data2[(alu8+9218)];
    var val21 = data2[(alu8+9219)];
    var val22 = data2[(alu8+9220)];
    var val23 = data2[(alu8+9221)];
    var val24 = data2[(alu8+9222)];
    var val25 = data2[(alu8+9223)];
    var val26 = data2[(alu8+9224)];
    var val27 = data2[(alu8+13824)];
    var val28 = data2[(alu8+13825)];
    var val29 = data2[(alu8+13826)];
    var val30 = data2[(alu8+13827)];
    var val31 = data2[(alu8+13828)];
    var val32 = data2[(alu8+13829)];
    var val33 = data2[(alu8+13830)];
    var val34 = data2[(alu8+13831)];
    var val35 = data2[(alu8+13832)];
    var alu9 = (alu0+alu2+(ridx0<<16)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-257)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-256)], alu5);
    var val39 = select(0.0f, data1[(alu9+-255)], alu5);
    var val40 = select(0.0f, data1[(alu9+-254)], alu5);
    var val41 = select(0.0f, data1[(alu9+-253)], alu5);
    var val42 = select(0.0f, data1[(alu9+-252)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+255)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+256)], alu6);
    var val50 = select(0.0f, data1[(alu9+257)], alu6);
    var val51 = select(0.0f, data1[(alu9+258)], alu6);
    var val52 = select(0.0f, data1[(alu9+259)], alu6);
    var val53 = select(0.0f, data1[(alu9+260)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data3[alu27];
  var val55 = data3[(alu27+1)];
  var val56 = data3[(alu27+2)];
  var val57 = data3[(alu27+3)];
  var alu28 = (alu0+(gidx2<<18)+alu1+alu2+alu3);
  data0[alu28] = (val54+acc0);
  data0[(alu28+1)] = (val54+acc4);
  data0[(alu28+2)] = (val54+acc8);
  data0[(alu28+3)] = (val54+acc12);
  data0[(alu28+65536)] = (val55+acc1);
  data0[(alu28+65537)] = (val55+acc5);
  data0[(alu28+65538)] = (val55+acc9);
  data0[(alu28+65539)] = (val55+acc13);
  data0[(alu28+131072)] = (val56+acc2);
  data0[(alu28+131073)] = (val56+acc6);
  data0[(alu28+131074)] = (val56+acc10);
  data0[(alu28+131075)] = (val56+acc14);
  data0[(alu28+196608)] = (val57+acc3);
  data0[(alu28+196609)] = (val57+acc7);
  data0[(alu28+196610)] = (val57+acc11);
  data0[(alu28+196611)] = (val57+acc15);
}`;

const r_256_32_512_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = ((gidx0<<16)+(lidx0<<11)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val3+val2+val1+val0);
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_256n6 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (acc0*1.9073486328125e-06f);
  }
}`;

const r_4_4_8_16_512_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = ((gidx0<<17)+(gidx1<<22)+(lidx0<<19)+(lidx1<<13)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+2048)];
    var val6 = data1[(alu0+2049)];
    var val7 = data1[(alu0+2050)];
    var val8 = data1[(alu0+2051)];
    var val9 = data1[(alu0+4096)];
    var val10 = data1[(alu0+4097)];
    var val11 = data1[(alu0+4098)];
    var val12 = data1[(alu0+4099)];
    var val13 = data1[(alu0+6144)];
    var val14 = data1[(alu0+6145)];
    var val15 = data1[(alu0+6146)];
    var val16 = data1[(alu0+6147)];
    var alu1 = (val2-val0);
    var alu2 = (val3-val0);
    var alu3 = (val4-val0);
    var alu4 = (val5-val0);
    var alu5 = (val6-val0);
    var alu6 = (val7-val0);
    var alu7 = (val8-val0);
    var alu8 = (val9-val0);
    var alu9 = (val10-val0);
    var alu10 = (val11-val0);
    var alu11 = (val12-val0);
    var alu12 = (val13-val0);
    var alu13 = (val14-val0);
    var alu14 = (val15-val0);
    var alu15 = (val16-val0);
    var alu16 = (val1-val0);
    acc0 = (acc0+(alu1*alu1)+(alu16*alu16)+(alu2*alu2)+(alu3*alu3));
    acc1 = (acc1+(alu4*alu4)+(alu5*alu5)+(alu6*alu6)+(alu7*alu7));
    acc2 = (acc2+(alu8*alu8)+(alu9*alu9)+(alu10*alu10)+(alu11*alu11));
    acc3 = (acc3+(alu12*alu12)+(alu13*alu13)+(alu14*alu14)+(alu15*alu15));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_32_256n7 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = sqrt((1/((acc0*1.9073486328125e-06f)+1e-05f)));
  }
}`;

const E_32_1024_8_16_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[gidx1];
  var val1 = data3[gidx1];
  var alu0 = (lidx0+(gidx1<<3));
  var val2 = data4[alu0];
  var val3 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<19)+(lidx0<<16)+(lidx1<<2));
  var val4 = data1[alu1];
  var alu2 = (alu1+1);
  var val5 = data1[alu2];
  var alu3 = (alu1+2);
  var val6 = data1[alu3];
  var alu4 = (alu1+3);
  var val7 = data1[alu4];
  var alu5 = -val3;
  var alu6 = (val2*val1*(val5-val0));
  data0[alu2] = ((1/(exp2(((alu5-alu6)*1.4426950408889634f))+1.0f))*(val3+alu6));
  var alu8 = (val2*val1*(val6-val0));
  data0[alu3] = ((1/(exp2(((alu5-alu8)*1.4426950408889634f))+1.0f))*(val3+alu8));
  var alu10 = (val2*val1*(val7-val0));
  data0[alu4] = ((1/(exp2(((alu5-alu10)*1.4426950408889634f))+1.0f))*(val3+alu10));
  var alu12 = (val2*val1*(val4-val0));
  data0[alu1] = ((1/(exp2(((alu5-alu12)*1.4426950408889634f))+1.0f))*(val3+alu12));
}`;

const r_64_32_4_8_16_256_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<11);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<8);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<255);
  var alu7 = ((lidx1+(gidx0<<4))<63);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu8 = ((gidx2*9216)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+2304)];
    var val10 = data2[(alu8+2305)];
    var val11 = data2[(alu8+2306)];
    var val12 = data2[(alu8+2307)];
    var val13 = data2[(alu8+2308)];
    var val14 = data2[(alu8+2309)];
    var val15 = data2[(alu8+2310)];
    var val16 = data2[(alu8+2311)];
    var val17 = data2[(alu8+2312)];
    var val18 = data2[(alu8+4608)];
    var val19 = data2[(alu8+4609)];
    var val20 = data2[(alu8+4610)];
    var val21 = data2[(alu8+4611)];
    var val22 = data2[(alu8+4612)];
    var val23 = data2[(alu8+4613)];
    var val24 = data2[(alu8+4614)];
    var val25 = data2[(alu8+4615)];
    var val26 = data2[(alu8+4616)];
    var val27 = data2[(alu8+6912)];
    var val28 = data2[(alu8+6913)];
    var val29 = data2[(alu8+6914)];
    var val30 = data2[(alu8+6915)];
    var val31 = data2[(alu8+6916)];
    var val32 = data2[(alu8+6917)];
    var val33 = data2[(alu8+6918)];
    var val34 = data2[(alu8+6919)];
    var val35 = data2[(alu8+6920)];
    var alu9 = (alu0+alu2+(ridx0<<16)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-257)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-256)], alu5);
    var val39 = select(0.0f, data1[(alu9+-255)], alu5);
    var val40 = select(0.0f, data1[(alu9+-254)], alu5);
    var val41 = select(0.0f, data1[(alu9+-253)], alu5);
    var val42 = select(0.0f, data1[(alu9+-252)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+255)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+256)], alu6);
    var val50 = select(0.0f, data1[(alu9+257)], alu6);
    var val51 = select(0.0f, data1[(alu9+258)], alu6);
    var val52 = select(0.0f, data1[(alu9+259)], alu6);
    var val53 = select(0.0f, data1[(alu9+260)], (alu6&alu7));
    acc0 = (acc0+(val3*val43)+(val0*val37)+(val6*val48)+(val1*val38)+(val4*val36)+(val7*val49)+(val2*val39)+(val5*val44)+(val8*val50));
    acc1 = (acc1+(val3*val36)+(val0*val38)+(val6*val49)+(val1*val39)+(val4*val44)+(val7*val50)+(val2*val40)+(val5*val45)+(val8*val51));
    acc2 = (acc2+(val3*val44)+(val0*val39)+(val6*val50)+(val1*val40)+(val4*val45)+(val7*val51)+(val2*val41)+(val5*val46)+(val8*val52));
    acc3 = (acc3+(val3*val45)+(val0*val40)+(val6*val51)+(val1*val41)+(val4*val46)+(val7*val52)+(val2*val42)+(val5*val47)+(val8*val53));
    acc4 = (acc4+(val9*val37)+(val12*val43)+(val15*val48)+(val10*val38)+(val13*val36)+(val16*val49)+(val11*val39)+(val14*val44)+(val17*val50));
    acc5 = (acc5+(val9*val38)+(val12*val36)+(val15*val49)+(val10*val39)+(val13*val44)+(val16*val50)+(val11*val40)+(val14*val45)+(val17*val51));
    acc6 = (acc6+(val9*val39)+(val12*val44)+(val15*val50)+(val10*val40)+(val13*val45)+(val16*val51)+(val11*val41)+(val14*val46)+(val17*val52));
    acc7 = (acc7+(val9*val40)+(val12*val45)+(val15*val51)+(val10*val41)+(val13*val46)+(val16*val52)+(val11*val42)+(val14*val47)+(val17*val53));
    acc8 = (acc8+(val18*val37)+(val21*val43)+(val24*val48)+(val19*val38)+(val22*val36)+(val25*val49)+(val20*val39)+(val23*val44)+(val26*val50));
    acc9 = (acc9+(val18*val38)+(val21*val36)+(val24*val49)+(val19*val39)+(val22*val44)+(val25*val50)+(val20*val40)+(val23*val45)+(val26*val51));
    acc10 = (acc10+(val18*val39)+(val21*val44)+(val24*val50)+(val19*val40)+(val22*val45)+(val25*val51)+(val20*val41)+(val23*val46)+(val26*val52));
    acc11 = (acc11+(val18*val40)+(val21*val45)+(val24*val51)+(val19*val41)+(val22*val46)+(val25*val52)+(val20*val42)+(val23*val47)+(val26*val53));
    acc12 = (acc12+(val27*val37)+(val30*val43)+(val33*val48)+(val28*val38)+(val31*val36)+(val34*val49)+(val29*val39)+(val32*val44)+(val35*val50));
    acc13 = (acc13+(val27*val38)+(val30*val36)+(val33*val49)+(val28*val39)+(val31*val44)+(val34*val50)+(val29*val40)+(val32*val45)+(val35*val51));
    acc14 = (acc14+(val27*val39)+(val30*val44)+(val33*val50)+(val28*val40)+(val31*val45)+(val34*val51)+(val29*val41)+(val32*val46)+(val35*val52));
    acc15 = (acc15+(val27*val40)+(val30*val45)+(val33*val51)+(val28*val41)+(val31*val46)+(val34*val52)+(val29*val42)+(val32*val47)+(val35*val53));
  }
  var alu27 = (alu0+(gidx2<<18)+alu1+alu2+alu3);
  data0[alu27] = acc0;
  data0[(alu27+1)] = acc1;
  data0[(alu27+2)] = acc2;
  data0[(alu27+3)] = acc3;
  data0[(alu27+65536)] = acc4;
  data0[(alu27+65537)] = acc5;
  data0[(alu27+65538)] = acc6;
  data0[(alu27+65539)] = acc7;
  data0[(alu27+131072)] = acc8;
  data0[(alu27+131073)] = acc9;
  data0[(alu27+131074)] = acc10;
  data0[(alu27+131075)] = acc11;
  data0[(alu27+196608)] = acc12;
  data0[(alu27+196609)] = acc13;
  data0[(alu27+196610)] = acc14;
  data0[(alu27+196611)] = acc15;
}`;

const r_8_1024_8_16_128_4_4_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu2 = ((gidx1<<14)+(lidx0<<11)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+512)];
    var val5 = data2[(alu2+513)];
    var val6 = data2[(alu2+514)];
    var val7 = data2[(alu2+515)];
    var val8 = data2[(alu2+1024)];
    var val9 = data2[(alu2+1025)];
    var val10 = data2[(alu2+1026)];
    var val11 = data2[(alu2+1027)];
    var val12 = data2[(alu2+1536)];
    var val13 = data2[(alu2+1537)];
    var val14 = data2[(alu2+1538)];
    var val15 = data2[(alu2+1539)];
    var alu3 = (alu0+alu1+(ridx0<<18));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+65536)];
    var val21 = data1[(alu3+65537)];
    var val22 = data1[(alu3+65538)];
    var val23 = data1[(alu3+65539)];
    var val24 = data1[(alu3+131072)];
    var val25 = data1[(alu3+131073)];
    var val26 = data1[(alu3+131074)];
    var val27 = data1[(alu3+131075)];
    var val28 = data1[(alu3+196608)];
    var val29 = data1[(alu3+196609)];
    var val30 = data1[(alu3+196610)];
    var val31 = data1[(alu3+196611)];
    acc0 = (acc0+(val20*val1)+(val16*val0)+(val24*val2)+(val28*val3));
    acc1 = (acc1+(val20*val5)+(val16*val4)+(val24*val6)+(val28*val7));
    acc2 = (acc2+(val20*val9)+(val16*val8)+(val24*val10)+(val28*val11));
    acc3 = (acc3+(val20*val13)+(val16*val12)+(val24*val14)+(val28*val15));
    acc4 = (acc4+(val17*val0)+(val21*val1)+(val25*val2)+(val29*val3));
    acc5 = (acc5+(val17*val4)+(val21*val5)+(val25*val6)+(val29*val7));
    acc6 = (acc6+(val17*val8)+(val21*val9)+(val25*val10)+(val29*val11));
    acc7 = (acc7+(val17*val12)+(val21*val13)+(val25*val14)+(val29*val15));
    acc8 = (acc8+(val18*val0)+(val22*val1)+(val26*val2)+(val30*val3));
    acc9 = (acc9+(val18*val4)+(val22*val5)+(val26*val6)+(val30*val7));
    acc10 = (acc10+(val18*val8)+(val22*val9)+(val26*val10)+(val30*val11));
    acc11 = (acc11+(val18*val12)+(val22*val13)+(val26*val14)+(val30*val15));
    acc12 = (acc12+(val19*val0)+(val23*val1)+(val27*val2)+(val31*val3));
    acc13 = (acc13+(val19*val4)+(val23*val5)+(val27*val6)+(val31*val7));
    acc14 = (acc14+(val19*val8)+(val23*val9)+(val27*val10)+(val31*val11));
    acc15 = (acc15+(val19*val12)+(val23*val13)+(val27*val14)+(val31*val15));
  }
  var alu21 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data5[alu21];
  var alu22 = (alu21+1);
  var val34 = data3[alu22];
  var val35 = data5[alu22];
  var alu23 = (alu21+2);
  var val36 = data3[alu23];
  var val37 = data5[alu23];
  var alu24 = (alu21+3);
  var val38 = data3[alu24];
  var val39 = data5[alu24];
  var alu25 = (alu0+(gidx1<<21)+(lidx0<<18)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+65536);
  var val44 = data4[alu29];
  var alu30 = (alu25+65537);
  var val45 = data4[alu30];
  var alu31 = (alu25+65538);
  var val46 = data4[alu31];
  var alu32 = (alu25+65539);
  var val47 = data4[alu32];
  var alu33 = (alu25+131072);
  var val48 = data4[alu33];
  var alu34 = (alu25+131073);
  var val49 = data4[alu34];
  var alu35 = (alu25+131074);
  var val50 = data4[alu35];
  var alu36 = (alu25+131075);
  var val51 = data4[alu36];
  var alu37 = (alu25+196608);
  var val52 = data4[alu37];
  var alu38 = (alu25+196609);
  var val53 = data4[alu38];
  var alu39 = (alu25+196610);
  var val54 = data4[alu39];
  var alu40 = (alu25+196611);
  var val55 = data4[alu40];
  data0[alu29] = (val34+acc1+val35+val44);
  data0[alu30] = (val34+acc5+val35+val45);
  data0[alu31] = (val34+acc9+val35+val46);
  data0[alu32] = (val34+acc13+val35+val47);
  data0[alu33] = (val36+acc2+val37+val48);
  data0[alu34] = (val36+acc6+val37+val49);
  data0[alu35] = (val36+acc10+val37+val50);
  data0[alu36] = (val36+acc14+val37+val51);
  data0[alu37] = (val38+acc3+val39+val52);
  data0[alu38] = (val38+acc7+val39+val53);
  data0[alu39] = (val38+acc11+val39+val54);
  data0[alu40] = (val38+acc15+val39+val55);
  data0[alu25] = (val32+acc0+val33+val40);
  data0[alu26] = (val32+acc4+val33+val41);
  data0[alu27] = (val32+acc8+val33+val42);
  data0[alu28] = (val32+acc12+val33+val43);
}`;

const r_64_32_4_8_16_256_4_4_3_3n1 = `enable f16;
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
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<11);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<8);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<255);
  var alu7 = ((lidx1+(gidx0<<4))<63);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu8 = ((gidx2*9216)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+2304)];
    var val10 = data2[(alu8+2305)];
    var val11 = data2[(alu8+2306)];
    var val12 = data2[(alu8+2307)];
    var val13 = data2[(alu8+2308)];
    var val14 = data2[(alu8+2309)];
    var val15 = data2[(alu8+2310)];
    var val16 = data2[(alu8+2311)];
    var val17 = data2[(alu8+2312)];
    var val18 = data2[(alu8+4608)];
    var val19 = data2[(alu8+4609)];
    var val20 = data2[(alu8+4610)];
    var val21 = data2[(alu8+4611)];
    var val22 = data2[(alu8+4612)];
    var val23 = data2[(alu8+4613)];
    var val24 = data2[(alu8+4614)];
    var val25 = data2[(alu8+4615)];
    var val26 = data2[(alu8+4616)];
    var val27 = data2[(alu8+6912)];
    var val28 = data2[(alu8+6913)];
    var val29 = data2[(alu8+6914)];
    var val30 = data2[(alu8+6915)];
    var val31 = data2[(alu8+6916)];
    var val32 = data2[(alu8+6917)];
    var val33 = data2[(alu8+6918)];
    var val34 = data2[(alu8+6919)];
    var val35 = data2[(alu8+6920)];
    var alu9 = (alu0+alu2+(ridx0<<16)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-257)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-256)], alu5);
    var val39 = select(0.0f, data1[(alu9+-255)], alu5);
    var val40 = select(0.0f, data1[(alu9+-254)], alu5);
    var val41 = select(0.0f, data1[(alu9+-253)], alu5);
    var val42 = select(0.0f, data1[(alu9+-252)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+255)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+256)], alu6);
    var val50 = select(0.0f, data1[(alu9+257)], alu6);
    var val51 = select(0.0f, data1[(alu9+258)], alu6);
    var val52 = select(0.0f, data1[(alu9+259)], alu6);
    var val53 = select(0.0f, data1[(alu9+260)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data3[alu27];
  var val55 = data3[(alu27+1)];
  var val56 = data3[(alu27+2)];
  var val57 = data3[(alu27+3)];
  var alu28 = (alu0+(gidx2<<18)+alu1+alu2+alu3);
  data0[alu28] = (val54+acc0);
  data0[(alu28+1)] = (val54+acc4);
  data0[(alu28+2)] = (val54+acc8);
  data0[(alu28+3)] = (val54+acc12);
  data0[(alu28+65536)] = (val55+acc1);
  data0[(alu28+65537)] = (val55+acc5);
  data0[(alu28+65538)] = (val55+acc9);
  data0[(alu28+65539)] = (val55+acc13);
  data0[(alu28+131072)] = (val56+acc2);
  data0[(alu28+131073)] = (val56+acc6);
  data0[(alu28+131074)] = (val56+acc10);
  data0[(alu28+131075)] = (val56+acc14);
  data0[(alu28+196608)] = (val57+acc3);
  data0[(alu28+196609)] = (val57+acc7);
  data0[(alu28+196610)] = (val57+acc11);
  data0[(alu28+196611)] = (val57+acc15);
}`;

const r_64_32_4_8_16_256_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<11);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<8);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<255);
  var alu7 = ((lidx1+(gidx0<<4))<63);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu8 = ((gidx2*9216)+(ridx0*9));
    var val0 = data3[alu8];
    var val1 = data3[(alu8+1)];
    var val2 = data3[(alu8+2)];
    var val3 = data3[(alu8+3)];
    var val4 = data3[(alu8+4)];
    var val5 = data3[(alu8+5)];
    var val6 = data3[(alu8+6)];
    var val7 = data3[(alu8+7)];
    var val8 = data3[(alu8+8)];
    var val9 = data3[(alu8+2304)];
    var val10 = data3[(alu8+2305)];
    var val11 = data3[(alu8+2306)];
    var val12 = data3[(alu8+2307)];
    var val13 = data3[(alu8+2308)];
    var val14 = data3[(alu8+2309)];
    var val15 = data3[(alu8+2310)];
    var val16 = data3[(alu8+2311)];
    var val17 = data3[(alu8+2312)];
    var val18 = data3[(alu8+4608)];
    var val19 = data3[(alu8+4609)];
    var val20 = data3[(alu8+4610)];
    var val21 = data3[(alu8+4611)];
    var val22 = data3[(alu8+4612)];
    var val23 = data3[(alu8+4613)];
    var val24 = data3[(alu8+4614)];
    var val25 = data3[(alu8+4615)];
    var val26 = data3[(alu8+4616)];
    var val27 = data3[(alu8+6912)];
    var val28 = data3[(alu8+6913)];
    var val29 = data3[(alu8+6914)];
    var val30 = data3[(alu8+6915)];
    var val31 = data3[(alu8+6916)];
    var val32 = data3[(alu8+6917)];
    var val33 = data3[(alu8+6918)];
    var val34 = data3[(alu8+6919)];
    var val35 = data3[(alu8+6920)];
    var alu9 = (alu0+alu2+(ridx0<<16)+alu1+alu3);
    var val36 = data2[alu9];
    var val37 = select(0.0f, data2[(alu9+-257)], (alu4&alu5));
    var val38 = select(0.0f, data2[(alu9+-256)], alu5);
    var val39 = select(0.0f, data2[(alu9+-255)], alu5);
    var val40 = select(0.0f, data2[(alu9+-254)], alu5);
    var val41 = select(0.0f, data2[(alu9+-253)], alu5);
    var val42 = select(0.0f, data2[(alu9+-252)], (alu7&alu5));
    var val43 = select(0.0f, data2[(alu9+-1)], alu4);
    var val44 = data2[(alu9+1)];
    var val45 = data2[(alu9+2)];
    var val46 = data2[(alu9+3)];
    var val47 = select(0.0f, data2[(alu9+4)], alu7);
    var val48 = select(0.0f, data2[(alu9+255)], (alu6&alu4));
    var val49 = select(0.0f, data2[(alu9+256)], alu6);
    var val50 = select(0.0f, data2[(alu9+257)], alu6);
    var val51 = select(0.0f, data2[(alu9+258)], alu6);
    var val52 = select(0.0f, data2[(alu9+259)], alu6);
    var val53 = select(0.0f, data2[(alu9+260)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data4[alu27];
  var val55 = data4[(alu27+1)];
  var val56 = data4[(alu27+2)];
  var val57 = data4[(alu27+3)];
  var alu28 = (alu0+(gidx2<<18)+alu1+alu2+alu3);
  var val58 = data1[alu28];
  var alu29 = (alu28+1);
  var val59 = data1[alu29];
  var alu30 = (alu28+2);
  var val60 = data1[alu30];
  var alu31 = (alu28+3);
  var val61 = data1[alu31];
  var alu32 = (alu28+65536);
  var val62 = data1[alu32];
  var alu33 = (alu28+65537);
  var val63 = data1[alu33];
  var alu34 = (alu28+65538);
  var val64 = data1[alu34];
  var alu35 = (alu28+65539);
  var val65 = data1[alu35];
  var alu36 = (alu28+131072);
  var val66 = data1[alu36];
  var alu37 = (alu28+131073);
  var val67 = data1[alu37];
  var alu38 = (alu28+131074);
  var val68 = data1[alu38];
  var alu39 = (alu28+131075);
  var val69 = data1[alu39];
  var alu40 = (alu28+196608);
  var val70 = data1[alu40];
  var alu41 = (alu28+196609);
  var val71 = data1[alu41];
  var alu42 = (alu28+196610);
  var val72 = data1[alu42];
  var alu43 = (alu28+196611);
  var val73 = data1[alu43];
  data0[alu33] = (val63+val55+acc5);
  data0[alu36] = (val66+val56+acc2);
  data0[alu32] = (val62+val55+acc1);
  data0[alu35] = (val65+val55+acc13);
  data0[alu40] = (val70+val57+acc3);
  data0[alu29] = (val59+val54+acc4);
  data0[alu30] = (val60+val54+acc8);
  data0[alu34] = (val64+val55+acc9);
  data0[alu38] = (val68+val56+acc10);
  data0[alu39] = (val69+val56+acc14);
  data0[alu37] = (val67+val56+acc6);
  data0[alu41] = (val71+val57+acc7);
  data0[alu42] = (val72+val57+acc11);
  data0[alu43] = (val73+val57+acc15);
  data0[alu28] = (val58+val54+acc0);
  data0[alu31] = (val61+val54+acc12);
}`;

const r_64_64_8_8_16_256_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+((gidx0+((lidx1+1)>>4))>>3));
  var alu1 = ((lidx0+1)>>1);
  var alu2 = (((gidx0+lidx1)<1)!=true);
  var alu3 = (((gidx1+lidx0)<1)!=true);
  var alu4 = ((lidx0+(gidx1<<3))<511);
  var alu5 = ((alu0+1)>>1);
  var alu6 = ((lidx1+(gidx0<<4))<127);
  var alu7 = ((gidx0<<5)+(lidx1<<1));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu8 = ((gidx2*9216)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+2304)];
    var val10 = data2[(alu8+2305)];
    var val11 = data2[(alu8+2306)];
    var val12 = data2[(alu8+2307)];
    var val13 = data2[(alu8+2308)];
    var val14 = data2[(alu8+2309)];
    var val15 = data2[(alu8+2310)];
    var val16 = data2[(alu8+2311)];
    var val17 = data2[(alu8+2312)];
    var val18 = data2[(alu8+4608)];
    var val19 = data2[(alu8+4609)];
    var val20 = data2[(alu8+4610)];
    var val21 = data2[(alu8+4611)];
    var val22 = data2[(alu8+4612)];
    var val23 = data2[(alu8+4613)];
    var val24 = data2[(alu8+4614)];
    var val25 = data2[(alu8+4615)];
    var val26 = data2[(alu8+4616)];
    var val27 = data2[(alu8+6912)];
    var val28 = data2[(alu8+6913)];
    var val29 = data2[(alu8+6914)];
    var val30 = data2[(alu8+6915)];
    var val31 = data2[(alu8+6916)];
    var val32 = data2[(alu8+6917)];
    var val33 = data2[(alu8+6918)];
    var val34 = data2[(alu8+6919)];
    var val35 = data2[(alu8+6920)];
    var alu9 = ((gidx1<<2)+(ridx0<<8));
    var alu10 = ((gidx1<<10)+(ridx0<<16));
    var alu11 = (alu10+((lidx0>>1)<<8)+alu7);
    var val36 = data1[alu11];
    var val37 = select(0.0f, data1[(alu11+-1)], alu2);
    var val38 = data1[(alu11+1)];
    var alu12 = (alu10+(alu1<<8)+alu7);
    var val39 = select(0.0f, data1[(alu12+-257)], (alu2&alu3));
    var val40 = select(0.0f, data1[(alu12+-256)], alu3);
    var val41 = select(0.0f, data1[(alu12+-255)], alu3);
    var val42 = select(0.0f, data1[(alu10+(alu5<<8)+alu7+-254)], (alu6&alu3));
    var val43 = select(0.0f, data1[(alu7+(((alu9+(alu0>>1))&65535)<<8)+2)], alu6);
    var alu13 = (alu7+(((alu9+alu1)&65535)<<8));
    var val44 = select(0.0f, data1[alu13], alu4);
    var val45 = select(0.0f, data1[(alu13+-1)], (alu4&alu2));
    var val46 = select(0.0f, data1[(alu13+1)], alu4);
    var val47 = select(0.0f, data1[(alu7+(((alu9+alu5)&65535)<<8)+2)], (alu4&alu6));
    var alu14 = (val38*val4);
    var alu15 = (val38*val5);
    var alu16 = (val38*val13);
    var alu17 = (val38*val14);
    var alu18 = (val38*val22);
    var alu19 = (val38*val23);
    var alu20 = (val38*val31);
    var alu21 = (val38*val32);
    var alu22 = (val40*val1);
    var alu23 = (val40*val10);
    var alu24 = (val40*val19);
    var alu25 = (val40*val28);
    var alu26 = (val41*val1);
    var alu27 = (val41*val2);
    var alu28 = (val41*val10);
    var alu29 = (val41*val11);
    var alu30 = (val41*val19);
    var alu31 = (val41*val20);
    var alu32 = (val41*val28);
    var alu33 = (val41*val29);
    var alu34 = (val46*val7);
    var alu35 = (val46*val8);
    var alu36 = (val46*val16);
    var alu37 = (val46*val17);
    var alu38 = (val46*val25);
    var alu39 = (val46*val26);
    var alu40 = (val46*val34);
    var alu41 = (val46*val35);
    var alu42 = (val36*val4);
    var alu43 = (val36*val13);
    var alu44 = (val36*val22);
    var alu45 = (val36*val31);
    var alu46 = ((val40*val0)+(val36*val3)+(val44*val6));
    var alu47 = (val44*val7);
    var alu48 = ((val40*val9)+(val36*val12)+(val44*val15));
    var alu49 = (val44*val16);
    var alu50 = ((val40*val18)+(val36*val21)+(val44*val24));
    var alu51 = (val44*val25);
    var alu52 = ((val40*val27)+(val36*val30)+(val44*val33));
    var alu53 = (val44*val34);
    acc0 = (acc0+(val37*val3)+(val39*val0)+(val45*val6)+alu22+alu42+alu47+(val40*val2)+(val36*val5)+(val44*val8));
    acc1 = (acc1+(val37*val12)+(val39*val9)+(val45*val15)+alu23+alu43+alu49+(val40*val11)+(val36*val14)+(val44*val17));
    acc2 = (acc2+(val37*val21)+(val39*val18)+(val45*val24)+alu24+alu44+alu51+(val40*val20)+(val36*val23)+(val44*val26));
    acc3 = (acc3+(val37*val30)+(val39*val27)+(val45*val33)+alu25+alu45+alu53+(val40*val29)+(val36*val32)+(val44*val35));
    acc4 = (acc4+alu46+alu22+alu42+alu47+alu27+alu15+alu35);
    acc5 = (acc5+alu48+alu23+alu43+alu49+alu29+alu17+alu37);
    acc6 = (acc6+alu50+alu24+alu44+alu51+alu31+alu19+alu39);
    acc7 = (acc7+alu52+alu25+alu45+alu53+alu33+alu21+alu41);
    acc8 = (acc8+alu46+alu26+alu14+alu34+alu27+alu15+alu35);
    acc9 = (acc9+alu48+alu28+alu16+alu36+alu29+alu17+alu37);
    acc10 = (acc10+alu50+alu30+alu18+alu38+alu31+alu19+alu39);
    acc11 = (acc11+alu52+alu32+alu20+alu40+alu33+alu21+alu41);
    acc12 = (acc12+(val38*val3)+(val41*val0)+(val46*val6)+alu26+alu14+alu34+(val42*val2)+(val43*val5)+(val47*val8));
    acc13 = (acc13+(val38*val12)+(val41*val9)+(val46*val15)+alu28+alu16+alu36+(val42*val11)+(val43*val14)+(val47*val17));
    acc14 = (acc14+(val38*val21)+(val41*val18)+(val46*val24)+alu30+alu18+alu38+(val42*val20)+(val43*val23)+(val47*val26));
    acc15 = (acc15+(val38*val30)+(val41*val27)+(val46*val33)+alu32+alu20+alu40+(val42*val29)+(val43*val32)+(val47*val35));
  }
  var alu71 = (gidx2<<2);
  var val48 = data3[alu71];
  var val49 = data3[(alu71+1)];
  var val50 = data3[(alu71+2)];
  var val51 = data3[(alu71+3)];
  var alu72 = ((gidx1<<12)+(gidx2<<20)+(gidx0<<6)+(lidx0<<9)+(lidx1<<2));
  data0[alu72] = (val48+acc0);
  data0[(alu72+1)] = (val48+acc4);
  data0[(alu72+2)] = (val48+acc8);
  data0[(alu72+3)] = (val48+acc12);
  data0[(alu72+262144)] = (val49+acc1);
  data0[(alu72+262145)] = (val49+acc5);
  data0[(alu72+262146)] = (val49+acc9);
  data0[(alu72+262147)] = (val49+acc13);
  data0[(alu72+524288)] = (val50+acc2);
  data0[(alu72+524289)] = (val50+acc6);
  data0[(alu72+524290)] = (val50+acc10);
  data0[(alu72+524291)] = (val50+acc14);
  data0[(alu72+786432)] = (val51+acc3);
  data0[(alu72+786433)] = (val51+acc7);
  data0[(alu72+786434)] = (val51+acc11);
  data0[(alu72+786435)] = (val51+acc15);
}`;

const r_256_32_2048_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 2048; ridx0++) {
    var alu0 = ((gidx0<<18)+(lidx0<<13)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+val3+val2+val1+val0);
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_256n8 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = (acc0*4.76837158203125e-07f);
  }
}`;

const r_4_4_8_16_2048_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 2048; ridx0++) {
    var alu0 = ((gidx0<<19)+(gidx1<<24)+(lidx0<<21)+(lidx1<<15)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+8192)];
    var val6 = data1[(alu0+8193)];
    var val7 = data1[(alu0+8194)];
    var val8 = data1[(alu0+8195)];
    var val9 = data1[(alu0+16384)];
    var val10 = data1[(alu0+16385)];
    var val11 = data1[(alu0+16386)];
    var val12 = data1[(alu0+16387)];
    var val13 = data1[(alu0+24576)];
    var val14 = data1[(alu0+24577)];
    var val15 = data1[(alu0+24578)];
    var val16 = data1[(alu0+24579)];
    var alu1 = (val2-val0);
    var alu2 = (val3-val0);
    var alu3 = (val4-val0);
    var alu4 = (val5-val0);
    var alu5 = (val6-val0);
    var alu6 = (val7-val0);
    var alu7 = (val8-val0);
    var alu8 = (val9-val0);
    var alu9 = (val10-val0);
    var alu10 = (val11-val0);
    var alu11 = (val12-val0);
    var alu12 = (val13-val0);
    var alu13 = (val14-val0);
    var alu14 = (val15-val0);
    var alu15 = (val16-val0);
    var alu16 = (val1-val0);
    acc0 = (acc0+(alu1*alu1)+(alu16*alu16)+(alu2*alu2)+(alu3*alu3));
    acc1 = (acc1+(alu4*alu4)+(alu5*alu5)+(alu6*alu6)+(alu7*alu7));
    acc2 = (acc2+(alu8*alu8)+(alu9*alu9)+(alu10*alu10)+(alu11*alu11));
    acc3 = (acc3+(alu12*alu12)+(alu13*alu13)+(alu14*alu14)+(alu15*alu15));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_32_256n9 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[(lidx0+(gidx0<<8))];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[gidx0] = sqrt((1/((acc0*4.76837158203125e-07f)+1e-05f)));
  }
}`;

const E_32_4096_8_16_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4096 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[gidx1];
  var val1 = data3[gidx1];
  var alu0 = (lidx0+(gidx1<<3));
  var val2 = data4[alu0];
  var val3 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<21)+(lidx0<<18)+(lidx1<<2));
  var val4 = data1[alu1];
  var alu2 = (alu1+1);
  var val5 = data1[alu2];
  var alu3 = (alu1+2);
  var val6 = data1[alu3];
  var alu4 = (alu1+3);
  var val7 = data1[alu4];
  var alu5 = -val3;
  var alu6 = (val2*val1*(val5-val0));
  data0[alu2] = ((1/(exp2(((alu5-alu6)*1.4426950408889634f))+1.0f))*(val3+alu6));
  var alu8 = (val2*val1*(val6-val0));
  data0[alu3] = ((1/(exp2(((alu5-alu8)*1.4426950408889634f))+1.0f))*(val3+alu8));
  var alu10 = (val2*val1*(val7-val0));
  data0[alu4] = ((1/(exp2(((alu5-alu10)*1.4426950408889634f))+1.0f))*(val3+alu10));
  var alu12 = (val2*val1*(val4-val0));
  data0[alu1] = ((1/(exp2(((alu5-alu12)*1.4426950408889634f))+1.0f))*(val3+alu12));
}`;

const r_32_64_8_8_16_256_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var gidx2 = i32(gindex.z); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<12);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<9);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<511);
  var alu7 = ((lidx1+(gidx0<<4))<127);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu8 = ((gidx2*9216)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+2304)];
    var val10 = data2[(alu8+2305)];
    var val11 = data2[(alu8+2306)];
    var val12 = data2[(alu8+2307)];
    var val13 = data2[(alu8+2308)];
    var val14 = data2[(alu8+2309)];
    var val15 = data2[(alu8+2310)];
    var val16 = data2[(alu8+2311)];
    var val17 = data2[(alu8+2312)];
    var val18 = data2[(alu8+4608)];
    var val19 = data2[(alu8+4609)];
    var val20 = data2[(alu8+4610)];
    var val21 = data2[(alu8+4611)];
    var val22 = data2[(alu8+4612)];
    var val23 = data2[(alu8+4613)];
    var val24 = data2[(alu8+4614)];
    var val25 = data2[(alu8+4615)];
    var val26 = data2[(alu8+4616)];
    var val27 = data2[(alu8+6912)];
    var val28 = data2[(alu8+6913)];
    var val29 = data2[(alu8+6914)];
    var val30 = data2[(alu8+6915)];
    var val31 = data2[(alu8+6916)];
    var val32 = data2[(alu8+6917)];
    var val33 = data2[(alu8+6918)];
    var val34 = data2[(alu8+6919)];
    var val35 = data2[(alu8+6920)];
    var alu9 = (alu0+alu2+(ridx0<<18)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-513)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-512)], alu5);
    var val39 = select(0.0f, data1[(alu9+-511)], alu5);
    var val40 = select(0.0f, data1[(alu9+-510)], alu5);
    var val41 = select(0.0f, data1[(alu9+-509)], alu5);
    var val42 = select(0.0f, data1[(alu9+-508)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+511)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+512)], alu6);
    var val50 = select(0.0f, data1[(alu9+513)], alu6);
    var val51 = select(0.0f, data1[(alu9+514)], alu6);
    var val52 = select(0.0f, data1[(alu9+515)], alu6);
    var val53 = select(0.0f, data1[(alu9+516)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data3[alu27];
  var val55 = data3[(alu27+1)];
  var val56 = data3[(alu27+2)];
  var val57 = data3[(alu27+3)];
  var alu28 = (alu0+(gidx2<<20)+alu1+alu2+alu3);
  data0[alu28] = (val54+acc0);
  data0[(alu28+1)] = (val54+acc4);
  data0[(alu28+2)] = (val54+acc8);
  data0[(alu28+3)] = (val54+acc12);
  data0[(alu28+262144)] = (val55+acc1);
  data0[(alu28+262145)] = (val55+acc5);
  data0[(alu28+262146)] = (val55+acc9);
  data0[(alu28+262147)] = (val55+acc13);
  data0[(alu28+524288)] = (val56+acc2);
  data0[(alu28+524289)] = (val56+acc6);
  data0[(alu28+524290)] = (val56+acc10);
  data0[(alu28+524291)] = (val56+acc14);
  data0[(alu28+786432)] = (val57+acc3);
  data0[(alu28+786433)] = (val57+acc7);
  data0[(alu28+786434)] = (val57+acc11);
  data0[(alu28+786435)] = (val57+acc15);
}`;

const E_16_4096_8_16_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4096 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<21)+(lidx0<<18)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = ((gidx1<<1)+(lidx0>>2));
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val7*(val3-val6));
  data0[alu2] = ((1/(exp2(((alu6-alu7)*1.4426950408889634f))+1.0f))*(val1+alu7));
  var alu9 = (val0*val7*(val4-val6));
  data0[alu3] = ((1/(exp2(((alu6-alu9)*1.4426950408889634f))+1.0f))*(val1+alu9));
  var alu11 = (val0*val7*(val5-val6));
  data0[alu4] = ((1/(exp2(((alu6-alu11)*1.4426950408889634f))+1.0f))*(val1+alu11));
  var alu13 = (val0*val7*(val2-val6));
  data0[alu1] = ((1/(exp2(((alu6-alu13)*1.4426950408889634f))+1.0f))*(val1+alu13));
}`;

const r_32_64_8_8_16_128_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var gidx2 = i32(gindex.z); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<12);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<9);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<511);
  var alu7 = ((lidx1+(gidx0<<4))<127);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu8 = ((gidx2*4608)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+1152)];
    var val10 = data2[(alu8+1153)];
    var val11 = data2[(alu8+1154)];
    var val12 = data2[(alu8+1155)];
    var val13 = data2[(alu8+1156)];
    var val14 = data2[(alu8+1157)];
    var val15 = data2[(alu8+1158)];
    var val16 = data2[(alu8+1159)];
    var val17 = data2[(alu8+1160)];
    var val18 = data2[(alu8+2304)];
    var val19 = data2[(alu8+2305)];
    var val20 = data2[(alu8+2306)];
    var val21 = data2[(alu8+2307)];
    var val22 = data2[(alu8+2308)];
    var val23 = data2[(alu8+2309)];
    var val24 = data2[(alu8+2310)];
    var val25 = data2[(alu8+2311)];
    var val26 = data2[(alu8+2312)];
    var val27 = data2[(alu8+3456)];
    var val28 = data2[(alu8+3457)];
    var val29 = data2[(alu8+3458)];
    var val30 = data2[(alu8+3459)];
    var val31 = data2[(alu8+3460)];
    var val32 = data2[(alu8+3461)];
    var val33 = data2[(alu8+3462)];
    var val34 = data2[(alu8+3463)];
    var val35 = data2[(alu8+3464)];
    var alu9 = (alu0+alu2+(ridx0<<18)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-513)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-512)], alu5);
    var val39 = select(0.0f, data1[(alu9+-511)], alu5);
    var val40 = select(0.0f, data1[(alu9+-510)], alu5);
    var val41 = select(0.0f, data1[(alu9+-509)], alu5);
    var val42 = select(0.0f, data1[(alu9+-508)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+511)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+512)], alu6);
    var val50 = select(0.0f, data1[(alu9+513)], alu6);
    var val51 = select(0.0f, data1[(alu9+514)], alu6);
    var val52 = select(0.0f, data1[(alu9+515)], alu6);
    var val53 = select(0.0f, data1[(alu9+516)], (alu6&alu7));
    acc0 = (acc0+(val3*val43)+(val0*val37)+(val6*val48)+(val1*val38)+(val4*val36)+(val7*val49)+(val2*val39)+(val5*val44)+(val8*val50));
    acc1 = (acc1+(val3*val36)+(val0*val38)+(val6*val49)+(val1*val39)+(val4*val44)+(val7*val50)+(val2*val40)+(val5*val45)+(val8*val51));
    acc2 = (acc2+(val3*val44)+(val0*val39)+(val6*val50)+(val1*val40)+(val4*val45)+(val7*val51)+(val2*val41)+(val5*val46)+(val8*val52));
    acc3 = (acc3+(val3*val45)+(val0*val40)+(val6*val51)+(val1*val41)+(val4*val46)+(val7*val52)+(val2*val42)+(val5*val47)+(val8*val53));
    acc4 = (acc4+(val9*val37)+(val12*val43)+(val15*val48)+(val10*val38)+(val13*val36)+(val16*val49)+(val11*val39)+(val14*val44)+(val17*val50));
    acc5 = (acc5+(val9*val38)+(val12*val36)+(val15*val49)+(val10*val39)+(val13*val44)+(val16*val50)+(val11*val40)+(val14*val45)+(val17*val51));
    acc6 = (acc6+(val9*val39)+(val12*val44)+(val15*val50)+(val10*val40)+(val13*val45)+(val16*val51)+(val11*val41)+(val14*val46)+(val17*val52));
    acc7 = (acc7+(val9*val40)+(val12*val45)+(val15*val51)+(val10*val41)+(val13*val46)+(val16*val52)+(val11*val42)+(val14*val47)+(val17*val53));
    acc8 = (acc8+(val18*val37)+(val21*val43)+(val24*val48)+(val19*val38)+(val22*val36)+(val25*val49)+(val20*val39)+(val23*val44)+(val26*val50));
    acc9 = (acc9+(val18*val38)+(val21*val36)+(val24*val49)+(val19*val39)+(val22*val44)+(val25*val50)+(val20*val40)+(val23*val45)+(val26*val51));
    acc10 = (acc10+(val18*val39)+(val21*val44)+(val24*val50)+(val19*val40)+(val22*val45)+(val25*val51)+(val20*val41)+(val23*val46)+(val26*val52));
    acc11 = (acc11+(val18*val40)+(val21*val45)+(val24*val51)+(val19*val41)+(val22*val46)+(val25*val52)+(val20*val42)+(val23*val47)+(val26*val53));
    acc12 = (acc12+(val27*val37)+(val30*val43)+(val33*val48)+(val28*val38)+(val31*val36)+(val34*val49)+(val29*val39)+(val32*val44)+(val35*val50));
    acc13 = (acc13+(val27*val38)+(val30*val36)+(val33*val49)+(val28*val39)+(val31*val44)+(val34*val50)+(val29*val40)+(val32*val45)+(val35*val51));
    acc14 = (acc14+(val27*val39)+(val30*val44)+(val33*val50)+(val28*val40)+(val31*val45)+(val34*val51)+(val29*val41)+(val32*val46)+(val35*val52));
    acc15 = (acc15+(val27*val40)+(val30*val45)+(val33*val51)+(val28*val41)+(val31*val46)+(val34*val52)+(val29*val42)+(val32*val47)+(val35*val53));
  }
  var alu27 = (alu0+(gidx2<<20)+alu1+alu2+alu3);
  data0[alu27] = acc0;
  data0[(alu27+1)] = acc1;
  data0[(alu27+2)] = acc2;
  data0[(alu27+3)] = acc3;
  data0[(alu27+262144)] = acc4;
  data0[(alu27+262145)] = acc5;
  data0[(alu27+262146)] = acc6;
  data0[(alu27+262147)] = acc7;
  data0[(alu27+524288)] = acc8;
  data0[(alu27+524289)] = acc9;
  data0[(alu27+524290)] = acc10;
  data0[(alu27+524291)] = acc11;
  data0[(alu27+786432)] = acc12;
  data0[(alu27+786433)] = acc13;
  data0[(alu27+786434)] = acc14;
  data0[(alu27+786435)] = acc15;
}`;

const r_4_4096_8_16_64_4_4_4 = `enable f16;
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
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4096 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu2 = ((gidx1<<13)+(lidx0<<10)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+256)];
    var val5 = data2[(alu2+257)];
    var val6 = data2[(alu2+258)];
    var val7 = data2[(alu2+259)];
    var val8 = data2[(alu2+512)];
    var val9 = data2[(alu2+513)];
    var val10 = data2[(alu2+514)];
    var val11 = data2[(alu2+515)];
    var val12 = data2[(alu2+768)];
    var val13 = data2[(alu2+769)];
    var val14 = data2[(alu2+770)];
    var val15 = data2[(alu2+771)];
    var alu3 = (alu0+alu1+(ridx0<<20));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+262144)];
    var val21 = data1[(alu3+262145)];
    var val22 = data1[(alu3+262146)];
    var val23 = data1[(alu3+262147)];
    var val24 = data1[(alu3+524288)];
    var val25 = data1[(alu3+524289)];
    var val26 = data1[(alu3+524290)];
    var val27 = data1[(alu3+524291)];
    var val28 = data1[(alu3+786432)];
    var val29 = data1[(alu3+786433)];
    var val30 = data1[(alu3+786434)];
    var val31 = data1[(alu3+786435)];
    acc0 = (acc0+(val20*val1)+(val16*val0)+(val24*val2)+(val28*val3));
    acc1 = (acc1+(val20*val5)+(val16*val4)+(val24*val6)+(val28*val7));
    acc2 = (acc2+(val20*val9)+(val16*val8)+(val24*val10)+(val28*val11));
    acc3 = (acc3+(val20*val13)+(val16*val12)+(val24*val14)+(val28*val15));
    acc4 = (acc4+(val17*val0)+(val21*val1)+(val25*val2)+(val29*val3));
    acc5 = (acc5+(val17*val4)+(val21*val5)+(val25*val6)+(val29*val7));
    acc6 = (acc6+(val17*val8)+(val21*val9)+(val25*val10)+(val29*val11));
    acc7 = (acc7+(val17*val12)+(val21*val13)+(val25*val14)+(val29*val15));
    acc8 = (acc8+(val18*val0)+(val22*val1)+(val26*val2)+(val30*val3));
    acc9 = (acc9+(val18*val4)+(val22*val5)+(val26*val6)+(val30*val7));
    acc10 = (acc10+(val18*val8)+(val22*val9)+(val26*val10)+(val30*val11));
    acc11 = (acc11+(val18*val12)+(val22*val13)+(val26*val14)+(val30*val15));
    acc12 = (acc12+(val19*val0)+(val23*val1)+(val27*val2)+(val31*val3));
    acc13 = (acc13+(val19*val4)+(val23*val5)+(val27*val6)+(val31*val7));
    acc14 = (acc14+(val19*val8)+(val23*val9)+(val27*val10)+(val31*val11));
    acc15 = (acc15+(val19*val12)+(val23*val13)+(val27*val14)+(val31*val15));
  }
  var alu21 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data5[alu21];
  var alu22 = (alu21+1);
  var val34 = data3[alu22];
  var val35 = data5[alu22];
  var alu23 = (alu21+2);
  var val36 = data3[alu23];
  var val37 = data5[alu23];
  var alu24 = (alu21+3);
  var val38 = data3[alu24];
  var val39 = data5[alu24];
  var alu25 = (alu0+(gidx1<<23)+(lidx0<<20)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+262144);
  var val44 = data4[alu29];
  var alu30 = (alu25+262145);
  var val45 = data4[alu30];
  var alu31 = (alu25+262146);
  var val46 = data4[alu31];
  var alu32 = (alu25+262147);
  var val47 = data4[alu32];
  var alu33 = (alu25+524288);
  var val48 = data4[alu33];
  var alu34 = (alu25+524289);
  var val49 = data4[alu34];
  var alu35 = (alu25+524290);
  var val50 = data4[alu35];
  var alu36 = (alu25+524291);
  var val51 = data4[alu36];
  var alu37 = (alu25+786432);
  var val52 = data4[alu37];
  var alu38 = (alu25+786433);
  var val53 = data4[alu38];
  var alu39 = (alu25+786434);
  var val54 = data4[alu39];
  var alu40 = (alu25+786435);
  var val55 = data4[alu40];
  data0[alu29] = (val34+acc1+val35+val44);
  data0[alu30] = (val34+acc5+val35+val45);
  data0[alu31] = (val34+acc9+val35+val46);
  data0[alu32] = (val34+acc13+val35+val47);
  data0[alu33] = (val36+acc2+val37+val48);
  data0[alu34] = (val36+acc6+val37+val49);
  data0[alu35] = (val36+acc10+val37+val50);
  data0[alu36] = (val36+acc14+val37+val51);
  data0[alu37] = (val38+acc3+val39+val52);
  data0[alu38] = (val38+acc7+val39+val53);
  data0[alu39] = (val38+acc11+val39+val54);
  data0[alu40] = (val38+acc15+val39+val55);
  data0[alu25] = (val32+acc0+val33+val40);
  data0[alu26] = (val32+acc4+val33+val41);
  data0[alu27] = (val32+acc8+val33+val42);
  data0[alu28] = (val32+acc12+val33+val43);
}`;

const r_32_64_8_8_16_128_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var gidx2 = i32(gindex.z); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<12);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<9);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<511);
  var alu7 = ((lidx1+(gidx0<<4))<127);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu8 = ((gidx2*4608)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+1152)];
    var val10 = data2[(alu8+1153)];
    var val11 = data2[(alu8+1154)];
    var val12 = data2[(alu8+1155)];
    var val13 = data2[(alu8+1156)];
    var val14 = data2[(alu8+1157)];
    var val15 = data2[(alu8+1158)];
    var val16 = data2[(alu8+1159)];
    var val17 = data2[(alu8+1160)];
    var val18 = data2[(alu8+2304)];
    var val19 = data2[(alu8+2305)];
    var val20 = data2[(alu8+2306)];
    var val21 = data2[(alu8+2307)];
    var val22 = data2[(alu8+2308)];
    var val23 = data2[(alu8+2309)];
    var val24 = data2[(alu8+2310)];
    var val25 = data2[(alu8+2311)];
    var val26 = data2[(alu8+2312)];
    var val27 = data2[(alu8+3456)];
    var val28 = data2[(alu8+3457)];
    var val29 = data2[(alu8+3458)];
    var val30 = data2[(alu8+3459)];
    var val31 = data2[(alu8+3460)];
    var val32 = data2[(alu8+3461)];
    var val33 = data2[(alu8+3462)];
    var val34 = data2[(alu8+3463)];
    var val35 = data2[(alu8+3464)];
    var alu9 = (alu0+alu2+(ridx0<<18)+alu1+alu3);
    var val36 = data1[alu9];
    var val37 = select(0.0f, data1[(alu9+-513)], (alu4&alu5));
    var val38 = select(0.0f, data1[(alu9+-512)], alu5);
    var val39 = select(0.0f, data1[(alu9+-511)], alu5);
    var val40 = select(0.0f, data1[(alu9+-510)], alu5);
    var val41 = select(0.0f, data1[(alu9+-509)], alu5);
    var val42 = select(0.0f, data1[(alu9+-508)], (alu7&alu5));
    var val43 = select(0.0f, data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select(0.0f, data1[(alu9+4)], alu7);
    var val48 = select(0.0f, data1[(alu9+511)], (alu6&alu4));
    var val49 = select(0.0f, data1[(alu9+512)], alu6);
    var val50 = select(0.0f, data1[(alu9+513)], alu6);
    var val51 = select(0.0f, data1[(alu9+514)], alu6);
    var val52 = select(0.0f, data1[(alu9+515)], alu6);
    var val53 = select(0.0f, data1[(alu9+516)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data3[alu27];
  var val55 = data3[(alu27+1)];
  var val56 = data3[(alu27+2)];
  var val57 = data3[(alu27+3)];
  var alu28 = (alu0+(gidx2<<20)+alu1+alu2+alu3);
  data0[alu28] = (val54+acc0);
  data0[(alu28+1)] = (val54+acc4);
  data0[(alu28+2)] = (val54+acc8);
  data0[(alu28+3)] = (val54+acc12);
  data0[(alu28+262144)] = (val55+acc1);
  data0[(alu28+262145)] = (val55+acc5);
  data0[(alu28+262146)] = (val55+acc9);
  data0[(alu28+262147)] = (val55+acc13);
  data0[(alu28+524288)] = (val56+acc2);
  data0[(alu28+524289)] = (val56+acc6);
  data0[(alu28+524290)] = (val56+acc10);
  data0[(alu28+524291)] = (val56+acc14);
  data0[(alu28+786432)] = (val57+acc3);
  data0[(alu28+786433)] = (val57+acc7);
  data0[(alu28+786434)] = (val57+acc11);
  data0[(alu28+786435)] = (val57+acc15);
}`;

const r_32_64_8_8_16_128_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var gidx2 = i32(gindex.z); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<12);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx0<<9);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<511);
  var alu7 = ((lidx1+(gidx0<<4))<127);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu8 = ((gidx2*4608)+(ridx0*9));
    var val0 = data3[alu8];
    var val1 = data3[(alu8+1)];
    var val2 = data3[(alu8+2)];
    var val3 = data3[(alu8+3)];
    var val4 = data3[(alu8+4)];
    var val5 = data3[(alu8+5)];
    var val6 = data3[(alu8+6)];
    var val7 = data3[(alu8+7)];
    var val8 = data3[(alu8+8)];
    var val9 = data3[(alu8+1152)];
    var val10 = data3[(alu8+1153)];
    var val11 = data3[(alu8+1154)];
    var val12 = data3[(alu8+1155)];
    var val13 = data3[(alu8+1156)];
    var val14 = data3[(alu8+1157)];
    var val15 = data3[(alu8+1158)];
    var val16 = data3[(alu8+1159)];
    var val17 = data3[(alu8+1160)];
    var val18 = data3[(alu8+2304)];
    var val19 = data3[(alu8+2305)];
    var val20 = data3[(alu8+2306)];
    var val21 = data3[(alu8+2307)];
    var val22 = data3[(alu8+2308)];
    var val23 = data3[(alu8+2309)];
    var val24 = data3[(alu8+2310)];
    var val25 = data3[(alu8+2311)];
    var val26 = data3[(alu8+2312)];
    var val27 = data3[(alu8+3456)];
    var val28 = data3[(alu8+3457)];
    var val29 = data3[(alu8+3458)];
    var val30 = data3[(alu8+3459)];
    var val31 = data3[(alu8+3460)];
    var val32 = data3[(alu8+3461)];
    var val33 = data3[(alu8+3462)];
    var val34 = data3[(alu8+3463)];
    var val35 = data3[(alu8+3464)];
    var alu9 = (alu0+alu2+(ridx0<<18)+alu1+alu3);
    var val36 = data2[alu9];
    var val37 = select(0.0f, data2[(alu9+-513)], (alu4&alu5));
    var val38 = select(0.0f, data2[(alu9+-512)], alu5);
    var val39 = select(0.0f, data2[(alu9+-511)], alu5);
    var val40 = select(0.0f, data2[(alu9+-510)], alu5);
    var val41 = select(0.0f, data2[(alu9+-509)], alu5);
    var val42 = select(0.0f, data2[(alu9+-508)], (alu7&alu5));
    var val43 = select(0.0f, data2[(alu9+-1)], alu4);
    var val44 = data2[(alu9+1)];
    var val45 = data2[(alu9+2)];
    var val46 = data2[(alu9+3)];
    var val47 = select(0.0f, data2[(alu9+4)], alu7);
    var val48 = select(0.0f, data2[(alu9+511)], (alu6&alu4));
    var val49 = select(0.0f, data2[(alu9+512)], alu6);
    var val50 = select(0.0f, data2[(alu9+513)], alu6);
    var val51 = select(0.0f, data2[(alu9+514)], alu6);
    var val52 = select(0.0f, data2[(alu9+515)], alu6);
    var val53 = select(0.0f, data2[(alu9+516)], (alu6&alu7));
    acc0 = (acc0+(val37*val0)+(val43*val3)+(val48*val6)+(val38*val1)+(val36*val4)+(val49*val7)+(val39*val2)+(val44*val5)+(val50*val8));
    acc1 = (acc1+(val37*val9)+(val43*val12)+(val48*val15)+(val38*val10)+(val36*val13)+(val49*val16)+(val39*val11)+(val44*val14)+(val50*val17));
    acc2 = (acc2+(val37*val18)+(val43*val21)+(val48*val24)+(val38*val19)+(val36*val22)+(val49*val25)+(val39*val20)+(val44*val23)+(val50*val26));
    acc3 = (acc3+(val37*val27)+(val43*val30)+(val48*val33)+(val38*val28)+(val36*val31)+(val49*val34)+(val39*val29)+(val44*val32)+(val50*val35));
    acc4 = (acc4+(val38*val0)+(val36*val3)+(val49*val6)+(val39*val1)+(val44*val4)+(val50*val7)+(val40*val2)+(val45*val5)+(val51*val8));
    acc5 = (acc5+(val38*val9)+(val36*val12)+(val49*val15)+(val39*val10)+(val44*val13)+(val50*val16)+(val40*val11)+(val45*val14)+(val51*val17));
    acc6 = (acc6+(val38*val18)+(val36*val21)+(val49*val24)+(val39*val19)+(val44*val22)+(val50*val25)+(val40*val20)+(val45*val23)+(val51*val26));
    acc7 = (acc7+(val38*val27)+(val36*val30)+(val49*val33)+(val39*val28)+(val44*val31)+(val50*val34)+(val40*val29)+(val45*val32)+(val51*val35));
    acc8 = (acc8+(val39*val0)+(val44*val3)+(val50*val6)+(val40*val1)+(val45*val4)+(val51*val7)+(val41*val2)+(val46*val5)+(val52*val8));
    acc9 = (acc9+(val39*val9)+(val44*val12)+(val50*val15)+(val40*val10)+(val45*val13)+(val51*val16)+(val41*val11)+(val46*val14)+(val52*val17));
    acc10 = (acc10+(val39*val18)+(val44*val21)+(val50*val24)+(val40*val19)+(val45*val22)+(val51*val25)+(val41*val20)+(val46*val23)+(val52*val26));
    acc11 = (acc11+(val39*val27)+(val44*val30)+(val50*val33)+(val40*val28)+(val45*val31)+(val51*val34)+(val41*val29)+(val46*val32)+(val52*val35));
    acc12 = (acc12+(val40*val0)+(val45*val3)+(val51*val6)+(val41*val1)+(val46*val4)+(val52*val7)+(val42*val2)+(val47*val5)+(val53*val8));
    acc13 = (acc13+(val40*val9)+(val45*val12)+(val51*val15)+(val41*val10)+(val46*val13)+(val52*val16)+(val42*val11)+(val47*val14)+(val53*val17));
    acc14 = (acc14+(val40*val18)+(val45*val21)+(val51*val24)+(val41*val19)+(val46*val22)+(val52*val25)+(val42*val20)+(val47*val23)+(val53*val26));
    acc15 = (acc15+(val40*val27)+(val45*val30)+(val51*val33)+(val41*val28)+(val46*val31)+(val52*val34)+(val42*val29)+(val47*val32)+(val53*val35));
  }
  var alu27 = (gidx2<<2);
  var val54 = data4[alu27];
  var val55 = data4[(alu27+1)];
  var val56 = data4[(alu27+2)];
  var val57 = data4[(alu27+3)];
  var alu28 = (alu0+(gidx2<<20)+alu1+alu2+alu3);
  var val58 = data1[alu28];
  var alu29 = (alu28+1);
  var val59 = data1[alu29];
  var alu30 = (alu28+2);
  var val60 = data1[alu30];
  var alu31 = (alu28+3);
  var val61 = data1[alu31];
  var alu32 = (alu28+262144);
  var val62 = data1[alu32];
  var alu33 = (alu28+262145);
  var val63 = data1[alu33];
  var alu34 = (alu28+262146);
  var val64 = data1[alu34];
  var alu35 = (alu28+262147);
  var val65 = data1[alu35];
  var alu36 = (alu28+524288);
  var val66 = data1[alu36];
  var alu37 = (alu28+524289);
  var val67 = data1[alu37];
  var alu38 = (alu28+524290);
  var val68 = data1[alu38];
  var alu39 = (alu28+524291);
  var val69 = data1[alu39];
  var alu40 = (alu28+786432);
  var val70 = data1[alu40];
  var alu41 = (alu28+786433);
  var val71 = data1[alu41];
  var alu42 = (alu28+786434);
  var val72 = data1[alu42];
  var alu43 = (alu28+786435);
  var val73 = data1[alu43];
  data0[alu33] = (val63+val55+acc5);
  data0[alu36] = (val66+val56+acc2);
  data0[alu32] = (val62+val55+acc1);
  data0[alu35] = (val65+val55+acc13);
  data0[alu40] = (val70+val57+acc3);
  data0[alu29] = (val59+val54+acc4);
  data0[alu30] = (val60+val54+acc8);
  data0[alu34] = (val64+val55+acc9);
  data0[alu38] = (val68+val56+acc10);
  data0[alu39] = (val69+val56+acc14);
  data0[alu37] = (val67+val56+acc6);
  data0[alu41] = (val71+val57+acc7);
  data0[alu42] = (val72+val57+acc11);
  data0[alu43] = (val73+val57+acc15);
  data0[alu28] = (val58+val54+acc0);
  data0[alu31] = (val61+val54+acc12);
}`;

const r_64_8_8_16_128_4_3_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (gidx1<<12);
  var alu2 = (lidx0<<9);
  var alu3 = (lidx1<<2);
  var alu4 = (((gidx0+lidx1)<1)!=true);
  var alu5 = (((gidx1+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx1<<3))<511);
  var alu7 = ((lidx1+(gidx0<<4))<127);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var alu8 = (ridx0*9);
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+1152)];
    var val10 = data2[(alu8+1153)];
    var val11 = data2[(alu8+1154)];
    var val12 = data2[(alu8+1155)];
    var val13 = data2[(alu8+1156)];
    var val14 = data2[(alu8+1157)];
    var val15 = data2[(alu8+1158)];
    var val16 = data2[(alu8+1159)];
    var val17 = data2[(alu8+1160)];
    var val18 = data2[(alu8+2304)];
    var val19 = data2[(alu8+2305)];
    var val20 = data2[(alu8+2306)];
    var val21 = data2[(alu8+2307)];
    var val22 = data2[(alu8+2308)];
    var val23 = data2[(alu8+2309)];
    var val24 = data2[(alu8+2310)];
    var val25 = data2[(alu8+2311)];
    var val26 = data2[(alu8+2312)];
    var alu9 = (alu1+alu2+(ridx0<<18)+alu0+alu3);
    var val27 = data1[alu9];
    var val28 = select(0.0f, data1[(alu9+-513)], (alu4&alu5));
    var val29 = select(0.0f, data1[(alu9+-512)], alu5);
    var val30 = select(0.0f, data1[(alu9+-511)], alu5);
    var val31 = select(0.0f, data1[(alu9+-510)], alu5);
    var val32 = select(0.0f, data1[(alu9+-509)], alu5);
    var val33 = select(0.0f, data1[(alu9+-508)], (alu7&alu5));
    var val34 = select(0.0f, data1[(alu9+-1)], alu4);
    var val35 = data1[(alu9+1)];
    var val36 = data1[(alu9+2)];
    var val37 = data1[(alu9+3)];
    var val38 = select(0.0f, data1[(alu9+4)], alu7);
    var val39 = select(0.0f, data1[(alu9+511)], (alu6&alu4));
    var val40 = select(0.0f, data1[(alu9+512)], alu6);
    var val41 = select(0.0f, data1[(alu9+513)], alu6);
    var val42 = select(0.0f, data1[(alu9+514)], alu6);
    var val43 = select(0.0f, data1[(alu9+515)], alu6);
    var val44 = select(0.0f, data1[(alu9+516)], (alu6&alu7));
    acc0 = (acc0+(val28*val0)+(val34*val3)+(val39*val6)+(val29*val1)+(val27*val4)+(val40*val7)+(val30*val2)+(val35*val5)+(val41*val8));
    acc1 = (acc1+(val28*val9)+(val34*val12)+(val39*val15)+(val29*val10)+(val27*val13)+(val40*val16)+(val30*val11)+(val35*val14)+(val41*val17));
    acc2 = (acc2+(val28*val18)+(val34*val21)+(val39*val24)+(val29*val19)+(val27*val22)+(val40*val25)+(val30*val20)+(val35*val23)+(val41*val26));
    acc3 = (acc3+(val29*val0)+(val27*val3)+(val40*val6)+(val30*val1)+(val35*val4)+(val41*val7)+(val31*val2)+(val36*val5)+(val42*val8));
    acc4 = (acc4+(val29*val9)+(val27*val12)+(val40*val15)+(val30*val10)+(val35*val13)+(val41*val16)+(val31*val11)+(val36*val14)+(val42*val17));
    acc5 = (acc5+(val29*val18)+(val27*val21)+(val40*val24)+(val30*val19)+(val35*val22)+(val41*val25)+(val31*val20)+(val36*val23)+(val42*val26));
    acc6 = (acc6+(val30*val0)+(val35*val3)+(val41*val6)+(val31*val1)+(val36*val4)+(val42*val7)+(val32*val2)+(val37*val5)+(val43*val8));
    acc7 = (acc7+(val30*val9)+(val35*val12)+(val41*val15)+(val31*val10)+(val36*val13)+(val42*val16)+(val32*val11)+(val37*val14)+(val43*val17));
    acc8 = (acc8+(val30*val18)+(val35*val21)+(val41*val24)+(val31*val19)+(val36*val22)+(val42*val25)+(val32*val20)+(val37*val23)+(val43*val26));
    acc9 = (acc9+(val31*val0)+(val36*val3)+(val42*val6)+(val32*val1)+(val37*val4)+(val43*val7)+(val33*val2)+(val38*val5)+(val44*val8));
    acc10 = (acc10+(val31*val9)+(val36*val12)+(val42*val15)+(val32*val10)+(val37*val13)+(val43*val16)+(val33*val11)+(val38*val14)+(val44*val17));
    acc11 = (acc11+(val31*val18)+(val36*val21)+(val42*val24)+(val32*val19)+(val37*val22)+(val43*val25)+(val33*val20)+(val38*val23)+(val44*val26));
  }
  var val45 = data3[0];
  var val46 = data3[1];
  var val47 = data3[2];
  var alu23 = (alu0+alu1+alu2+alu3);
  data0[alu23] = ((val45+acc0+1.0f)*0.5f);
  data0[(alu23+1)] = ((val45+acc3+1.0f)*0.5f);
  data0[(alu23+2)] = ((val45+acc6+1.0f)*0.5f);
  data0[(alu23+3)] = ((val45+acc9+1.0f)*0.5f);
  data0[(alu23+262144)] = ((val46+acc1+1.0f)*0.5f);
  data0[(alu23+262145)] = ((val46+acc4+1.0f)*0.5f);
  data0[(alu23+262146)] = ((val46+acc7+1.0f)*0.5f);
  data0[(alu23+262147)] = ((val46+acc10+1.0f)*0.5f);
  data0[(alu23+524288)] = ((val47+acc2+1.0f)*0.5f);
  data0[(alu23+524289)] = ((val47+acc5+1.0f)*0.5f);
  data0[(alu23+524290)] = ((val47+acc8+1.0f)*0.5f);
  data0[(alu23+524291)] = ((val47+acc11+1.0f)*0.5f);
}`;

const E_2048_32_3_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<atomic<u32>>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 32 */
  var lidx1 = i32(lindex.y); /* 3 */
  var alu0 = (lidx1+(gidx0*384)+(lidx0*12));
  var alu1 = (alu0+3);
  var alu2 = (alu0+6);
  var alu3 = (alu0+9);
  var alu4 = ((gidx0<<7)+(lidx0<<2)+(lidx1<<18));
  var val0 = data1[alu4];
  var val1 = data1[(alu4+1)];
  var val2 = data1[(alu4+2)];
  var val3 = data1[(alu4+3)];
  var val4 = atomicLoad(&data0[(alu0>>2)]);
  var val5 = atomicLoad(&data0[(alu1>>2)]);
  var val6 = atomicLoad(&data0[(alu2>>2)]);
  var val7 = atomicLoad(&data0[(alu3>>2)]);
  var alu5 = (((u32(alu0))&3u)<<3u);
  var alu6 = (((u32(alu1))&3u)<<3u);
  var alu7 = (((u32(alu2))&3u)<<3u);
  var alu8 = (((u32(alu3))&3u)<<3u);
  var alu9 = select(select(val1,(val1*0.5f),((bool(val1))!=true)),0.0f,(val1<0.0f));
  var alu10 = -alu9;
  var alu11 = select(select(val2,(val2*0.5f),((bool(val2))!=true)),0.0f,(val2<0.0f));
  var alu12 = -alu11;
  var alu13 = select(select(val3,(val3*0.5f),((bool(val3))!=true)),0.0f,(val3<0.0f));
  var alu14 = -alu13;
  var alu15 = select(select(val0,(val0*0.5f),((bool(val0))!=true)),0.0f,(val0<0.0f));
  var alu16 = -alu15;
  atomicAnd(&data0[(alu1>>2)],((255u<<alu6)^4294967295u));
  atomicAdd(&data0[(alu1>>2)],((u32(((u32((i32((select(select(alu10,((alu9*-0.5f)+-0.5f),((alu10!=-1.0f)!=true)),-1.0f,(alu10<-1.0f))*-255.0f)))))&255u)))<<alu6));
  atomicAnd(&data0[(alu2>>2)],((255u<<alu7)^4294967295u));
  atomicAdd(&data0[(alu2>>2)],((u32(((u32((i32((select(select(alu12,((alu11*-0.5f)+-0.5f),((alu12!=-1.0f)!=true)),-1.0f,(alu12<-1.0f))*-255.0f)))))&255u)))<<alu7));
  atomicAnd(&data0[(alu3>>2)],((255u<<alu8)^4294967295u));
  atomicAdd(&data0[(alu3>>2)],((u32(((u32((i32((select(select(alu14,((alu13*-0.5f)+-0.5f),((alu14!=-1.0f)!=true)),-1.0f,(alu14<-1.0f))*-255.0f)))))&255u)))<<alu8));
  atomicAnd(&data0[(alu0>>2)],((255u<<alu5)^4294967295u));
  atomicAdd(&data0[(alu0>>2)],((u32(((u32((i32((select(select(alu16,((alu15*-0.5f)+-0.5f),((alu16!=-1.0f)!=true)),-1.0f,(alu16<-1.0f))*-255.0f)))))&255u)))<<alu5));
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 65536);;
    const input0 = createEmptyBuf(device, 65536);;
    const buf_1 = createEmptyBuf(device, 65536);;
    const buf_2 = createWeightBuf(device, 64, getTensorBuffer(safetensor, metadata['first_stage_model.post_quant_conv.weight']));
    const buf_3 = createWeightBuf(device, 16, getTensorBuffer(safetensor, metadata['first_stage_model.post_quant_conv.bias']));
    const buf_4 = createEmptyBuf(device, 8388608);;
    const buf_5 = createWeightBuf(device, 73728, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.conv_in.weight']));
    const buf_6 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.conv_in.bias']));
    const buf_7 = createEmptyBuf(device, 32768);;
    const buf_8 = createEmptyBuf(device, 128);;
    const buf_9 = createEmptyBuf(device, 128);;
    const buf_10 = createEmptyBuf(device, 8388608);;
    const buf_11 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.norm1.weight']));
    const buf_12 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.norm1.bias']));
    const buf_13 = createEmptyBuf(device, 8388608);;
    const buf_14 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.conv1.weight']));
    const buf_15 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.conv1.bias']));
    const buf_16 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.norm2.weight']));
    const buf_17 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.norm2.bias']));
    const buf_18 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.conv2.weight']));
    const buf_19 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_1.conv2.bias']));
    const buf_20 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.norm.weight']));
    const buf_21 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.norm.bias']));
    const buf_22 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.q.weight']));
    const buf_23 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.q.bias']));
    const buf_24 = createEmptyBuf(device, 8388608);;
    const buf_25 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.k.weight']));
    const buf_26 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.k.bias']));
    const buf_27 = createEmptyBuf(device, 8388608);;
    const buf_28 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.v.weight']));
    const buf_29 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.v.bias']));
    const buf_30 = createEmptyBuf(device, 67108864);;
    const buf_31 = createEmptyBuf(device, 16384);;
    const buf_32 = createEmptyBuf(device, 16384);;
    const buf_33 = createEmptyBuf(device, 67108864);;
    const buf_34 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.proj_out.weight']));
    const buf_35 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.attn_1.proj_out.bias']));
    const buf_36 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.norm1.weight']));
    const buf_37 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.norm1.bias']));
    const buf_38 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.conv1.weight']));
    const buf_39 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.conv1.bias']));
    const buf_40 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.norm2.weight']));
    const buf_41 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.norm2.bias']));
    const buf_42 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.conv2.weight']));
    const buf_43 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.mid.block_2.conv2.bias']));
    const buf_44 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.norm1.weight']));
    const buf_45 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.norm1.bias']));
    const buf_46 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.conv1.weight']));
    const buf_47 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.conv1.bias']));
    const buf_48 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.norm2.weight']));
    const buf_49 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.norm2.bias']));
    const buf_50 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.conv2.weight']));
    const buf_51 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.0.conv2.bias']));
    const buf_52 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.norm1.weight']));
    const buf_53 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.norm1.bias']));
    const buf_54 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.conv1.weight']));
    const buf_55 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.conv1.bias']));
    const buf_56 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.norm2.weight']));
    const buf_57 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.norm2.bias']));
    const buf_58 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.conv2.weight']));
    const buf_59 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.1.conv2.bias']));
    const buf_60 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.norm1.weight']));
    const buf_61 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.norm1.bias']));
    const buf_62 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.conv1.weight']));
    const buf_63 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.conv1.bias']));
    const buf_64 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.norm2.weight']));
    const buf_65 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.norm2.bias']));
    const buf_66 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.conv2.weight']));
    const buf_67 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.block.2.conv2.bias']));
    const buf_68 = createEmptyBuf(device, 33554432);;
    const buf_69 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.upsample.conv.weight']));
    const buf_70 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.3.upsample.conv.bias']));
    const buf_71 = createEmptyBuf(device, 33554432);;
    const buf_72 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.norm1.weight']));
    const buf_73 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.norm1.bias']));
    const buf_74 = createEmptyBuf(device, 33554432);;
    const buf_75 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.conv1.weight']));
    const buf_76 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.conv1.bias']));
    const buf_77 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.norm2.weight']));
    const buf_78 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.norm2.bias']));
    const buf_79 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.conv2.weight']));
    const buf_80 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.0.conv2.bias']));
    const buf_81 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.norm1.weight']));
    const buf_82 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.norm1.bias']));
    const buf_83 = createEmptyBuf(device, 33554432);;
    const buf_84 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.conv1.weight']));
    const buf_85 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.conv1.bias']));
    const buf_86 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.norm2.weight']));
    const buf_87 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.norm2.bias']));
    const buf_88 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.conv2.weight']));
    const buf_89 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.1.conv2.bias']));
    const buf_90 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.norm1.weight']));
    const buf_91 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.norm1.bias']));
    const buf_92 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.conv1.weight']));
    const buf_93 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.conv1.bias']));
    const buf_94 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.norm2.weight']));
    const buf_95 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.norm2.bias']));
    const buf_96 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.conv2.weight']));
    const buf_97 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.block.2.conv2.bias']));
    const buf_98 = createEmptyBuf(device, 134217728);;
    const buf_99 = createWeightBuf(device, 9437184, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.upsample.conv.weight']));
    const buf_100 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.2.upsample.conv.bias']));
    const buf_101 = createEmptyBuf(device, 134217728);;
    const buf_102 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.norm1.weight']));
    const buf_103 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.norm1.bias']));
    const buf_104 = createWeightBuf(device, 4718592, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.conv1.weight']));
    const buf_105 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.conv1.bias']));
    const buf_106 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.norm2.weight']));
    const buf_107 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.norm2.bias']));
    const buf_108 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.conv2.weight']));
    const buf_109 = createWeightBuf(device, 524288, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.nin_shortcut.weight']));
    const buf_110 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.nin_shortcut.bias']));
    const buf_111 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.0.conv2.bias']));
    const buf_112 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.norm1.weight']));
    const buf_113 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.norm1.bias']));
    const buf_114 = createEmptyBuf(device, 67108864);;
    const buf_115 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.conv1.weight']));
    const buf_116 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.conv1.bias']));
    const buf_117 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.norm2.weight']));
    const buf_118 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.norm2.bias']));
    const buf_119 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.conv2.weight']));
    const buf_120 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.1.conv2.bias']));
    const buf_121 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.norm1.weight']));
    const buf_122 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.norm1.bias']));
    const buf_123 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.conv1.weight']));
    const buf_124 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.conv1.bias']));
    const buf_125 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.norm2.weight']));
    const buf_126 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.norm2.bias']));
    const buf_127 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.conv2.weight']));
    const buf_128 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.block.2.conv2.bias']));
    const buf_129 = createEmptyBuf(device, 268435456);;
    const buf_130 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.upsample.conv.weight']));
    const buf_131 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.1.upsample.conv.bias']));
    const buf_132 = createEmptyBuf(device, 268435456);;
    const buf_133 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.norm1.weight']));
    const buf_134 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.norm1.bias']));
    const buf_135 = createWeightBuf(device, 1179648, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.conv1.weight']));
    const buf_136 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.conv1.bias']));
    const buf_137 = createEmptyBuf(device, 134217728);;
    const buf_138 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.norm2.weight']));
    const buf_139 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.norm2.bias']));
    const buf_140 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.conv2.weight']));
    const buf_141 = createWeightBuf(device, 131072, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.nin_shortcut.weight']));
    const buf_142 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.nin_shortcut.bias']));
    const buf_143 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.0.conv2.bias']));
    const buf_144 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.norm1.weight']));
    const buf_145 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.norm1.bias']));
    const buf_146 = createEmptyBuf(device, 134217728);;
    const buf_147 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.conv1.weight']));
    const buf_148 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.conv1.bias']));
    const buf_149 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.norm2.weight']));
    const buf_150 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.norm2.bias']));
    const buf_151 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.conv2.weight']));
    const buf_152 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.1.conv2.bias']));
    const buf_153 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.norm1.weight']));
    const buf_154 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.norm1.bias']));
    const buf_155 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.conv1.weight']));
    const buf_156 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.conv1.bias']));
    const buf_157 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.norm2.weight']));
    const buf_158 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.norm2.bias']));
    const buf_159 = createEmptyBuf(device, 134217728);;
    const buf_160 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.conv2.weight']));
    const buf_161 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.up.0.block.2.conv2.bias']));
    const buf_162 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.norm_out.weight']));
    const buf_163 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.norm_out.bias']));
    const buf_164 = createEmptyBuf(device, 3145728);;
    const buf_165 = createWeightBuf(device, 13824, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.conv_out.weight']));
    const buf_166 = createWeightBuf(device, 12, getTensorBuffer(safetensor, metadata['first_stage_model.decoder.conv_out.bias']));
    const output0 = createEmptyBuf(device, 786432);;

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [E_128_32_4n4, r_32_32_4_4_4, r_128_8_8_16_4_4_4_3_3, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3n1, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4n1, r_16_64_8_16_128_4_4_4, r_16_64_8_16_128_4_4_4, r_16_64_8_16_128_4_4_4, r_128_64_8_16_128_4_4_4, r_128_32_1024_4, r_32_32_1024_4_4, E_512_64_8_16_4, r_128_8_8_16_1024_4_4_4, r_16_64_8_16_128_4_4_4n1, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3n1, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3n1, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3n1, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3, r_256_32_64_4, r_32_256, r_4_4_8_16_64_4_4, r_32_256n1, E_64_64_8_16_4, r_128_8_8_16_512_4_4_3_3n1, r_128_16_2_8_16_512_4_4_3_3, r_256_32_256_4, r_32_256n2, r_4_4_8_16_256_4_4, r_32_256n3, E_64_256_8_16_4, r_128_16_2_8_16_512_4_4_3_3n1, r_256_32_256_4, r_32_256n2, r_4_4_8_16_256_4_4, r_32_256n3, E_64_256_8_16_4, r_128_16_2_8_16_512_4_4_3_3n2, r_256_32_256_4, r_32_256n2, r_4_4_8_16_256_4_4, r_32_256n3, E_64_256_8_16_4, r_128_16_2_8_16_512_4_4_3_3n1, r_256_32_256_4, r_32_256n2, r_4_4_8_16_256_4_4, r_32_256n3, E_64_256_8_16_4, r_128_16_2_8_16_512_4_4_3_3n2, r_256_32_256_4, r_32_256n2, r_4_4_8_16_256_4_4, r_32_256n3, E_64_256_8_16_4, r_128_16_2_8_16_512_4_4_3_3n1, r_256_32_256_4, r_32_256n2, r_4_4_8_16_256_4_4, r_32_256n3, E_64_256_8_16_4, r_128_16_2_8_16_512_4_4_3_3n2, r_128_32_4_8_16_512_4_4_3_3, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_64_1024_8_16_4, r_64_32_4_8_16_512_4_4_3_3, r_256_32_512_4, r_32_256n6, r_4_4_8_16_512_4_4, r_32_256n7, E_32_1024_8_16_4, r_64_32_4_8_16_256_4_4_3_3, r_8_1024_8_16_128_4_4_4, r_256_32_512_4, r_32_256n6, r_4_4_8_16_512_4_4, r_32_256n7, E_32_1024_8_16_4, r_64_32_4_8_16_256_4_4_3_3n1, r_256_32_512_4, r_32_256n6, r_4_4_8_16_512_4_4, r_32_256n7, E_32_1024_8_16_4, r_64_32_4_8_16_256_4_4_3_3n2, r_256_32_512_4, r_32_256n6, r_4_4_8_16_512_4_4, r_32_256n7, E_32_1024_8_16_4, r_64_32_4_8_16_256_4_4_3_3n1, r_256_32_512_4, r_32_256n6, r_4_4_8_16_512_4_4, r_32_256n7, E_32_1024_8_16_4, r_64_32_4_8_16_256_4_4_3_3n2, r_64_64_8_8_16_256_4_4_3_3, r_256_32_2048_4, r_32_256n8, r_4_4_8_16_2048_4_4, r_32_256n9, E_32_4096_8_16_4, r_32_64_8_8_16_256_4_4_3_3, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_16_4096_8_16_4, r_32_64_8_8_16_128_4_4_3_3, r_4_4096_8_16_64_4_4_4, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_16_4096_8_16_4, r_32_64_8_8_16_128_4_4_3_3n1, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_16_4096_8_16_4, r_32_64_8_8_16_128_4_4_3_3n2, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_16_4096_8_16_4, r_32_64_8_8_16_128_4_4_3_3n1, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_16_4096_8_16_4, r_32_64_8_8_16_128_4_4_3_3n2, r_256_32_1024_4, r_32_256n4, r_4_4_8_16_1024_4_4, r_32_256n5, E_16_4096_8_16_4, r_64_8_8_16_128_4_3_3_3, E_2048_32_3_4];
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
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_1, buf_0, buf_2, buf_3], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_4, buf_1, buf_5, buf_6], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_7, buf_4], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_7, buf_4, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_10, buf_4, buf_8, buf_9, buf_11, buf_12], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_13, buf_10, buf_14, buf_15], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_7, buf_13], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_7, buf_13, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_10, buf_13, buf_8, buf_9, buf_16, buf_17], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_13, buf_4, buf_10, buf_18, buf_19], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_7, buf_13], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_7, buf_13, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_4, buf_13, buf_8, buf_9, buf_20, buf_21], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_10, buf_4, buf_22, buf_23], [64, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_24, buf_4, buf_25, buf_26], [64, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_27, buf_4, buf_28, buf_29], [64, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_30, buf_10, buf_24], [64, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_31, buf_30], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_32, buf_30, buf_31], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_33, buf_30, buf_31, buf_32], [64, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_4, buf_33, buf_27], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_10, buf_13, buf_4, buf_34, buf_35], [64, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_7, buf_10], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_7, buf_10, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_13, buf_10, buf_8, buf_9, buf_36, buf_37], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_24, buf_13, buf_38, buf_39], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_7, buf_24], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_7, buf_24, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_27, buf_24, buf_8, buf_9, buf_40, buf_41], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_4, buf_10, buf_27, buf_42, buf_43], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_7, buf_4], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_7, buf_4, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_10, buf_4, buf_8, buf_9, buf_44, buf_45], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_13, buf_10, buf_46, buf_47], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_7, buf_13], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_7, buf_13, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_24, buf_13, buf_8, buf_9, buf_48, buf_49], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_27, buf_4, buf_24, buf_50, buf_51], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_7, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_7, buf_27, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_4, buf_27, buf_8, buf_9, buf_52, buf_53], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_10, buf_4, buf_54, buf_55], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_7, buf_10], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_7, buf_10, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_13, buf_10, buf_8, buf_9, buf_56, buf_57], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_24, buf_27, buf_13, buf_58, buf_59], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_7, buf_24], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_7, buf_24, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_27, buf_24, buf_8, buf_9, buf_60, buf_61], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_4, buf_27, buf_62, buf_63], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_7, buf_4], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_7, buf_4, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_10, buf_4, buf_8, buf_9, buf_64, buf_65], [64, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_13, buf_24, buf_10, buf_66, buf_67], [8, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_68, buf_13, buf_69, buf_70], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_7, buf_68], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_7, buf_68, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_71, buf_68, buf_8, buf_9, buf_72, buf_73], [256, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_74, buf_71, buf_75, buf_76], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_7, buf_74], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_7, buf_74, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_71, buf_74, buf_8, buf_9, buf_77, buf_78], [256, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_74, buf_68, buf_71, buf_79, buf_80], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_7, buf_74], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_7, buf_74, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_71, buf_74, buf_8, buf_9, buf_81, buf_82], [256, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_83, buf_71, buf_84, buf_85], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_7, buf_83], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_7, buf_83, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_71, buf_83, buf_8, buf_9, buf_86, buf_87], [256, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_83, buf_74, buf_71, buf_88, buf_89], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_7, buf_83], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_7, buf_83, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_74, buf_83, buf_8, buf_9, buf_90, buf_91], [256, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_71, buf_74, buf_92, buf_93], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_7, buf_71], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_7, buf_71, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_74, buf_71, buf_8, buf_9, buf_94, buf_95], [256, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_71, buf_83, buf_74, buf_96, buf_97], [2, 16, 128]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_98, buf_71, buf_99, buf_100], [4, 32, 128]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_7, buf_98], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_7, buf_98, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_101, buf_98, buf_8, buf_9, buf_102, buf_103], [1024, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_30, buf_101, buf_104, buf_105], [4, 32, 64]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_7, buf_30], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_7, buf_30, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_33, buf_30, buf_8, buf_9, buf_106, buf_107], [1024, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_30, buf_33, buf_108], [4, 32, 64]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_33, buf_98, buf_109, buf_110, buf_30, buf_111], [1024, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_7, buf_33], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_7, buf_33, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_30, buf_33, buf_8, buf_9, buf_112, buf_113], [1024, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_114, buf_30, buf_115, buf_116], [4, 32, 64]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_7, buf_114], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_7, buf_114, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_30, buf_114, buf_8, buf_9, buf_117, buf_118], [1024, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_114, buf_33, buf_30, buf_119, buf_120], [4, 32, 64]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_7, buf_114], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_7, buf_114, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_33, buf_114, buf_8, buf_9, buf_121, buf_122], [1024, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_30, buf_33, buf_123, buf_124], [4, 32, 64]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_7, buf_30], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_7, buf_30, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_33, buf_30, buf_8, buf_9, buf_125, buf_126], [1024, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_30, buf_114, buf_33, buf_127, buf_128], [4, 32, 64]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_129, buf_30, buf_130, buf_131], [8, 64, 64]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_7, buf_129], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_7, buf_129, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_132, buf_129, buf_8, buf_9, buf_133, buf_134], [4096, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_101, buf_132, buf_135, buf_136], [8, 64, 32]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_7, buf_101], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_7, buf_101, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_137, buf_101, buf_8, buf_9, buf_138, buf_139], [4096, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_101, buf_137, buf_140], [8, 64, 32]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_137, buf_129, buf_141, buf_142, buf_101, buf_143], [4096, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_7, buf_137], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_7, buf_137, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [buf_101, buf_137, buf_8, buf_9, buf_144, buf_145], [4096, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[171], layouts[171], infinityBuf, [buf_146, buf_101, buf_147, buf_148], [8, 64, 32]);
        addComputePass(device, commandEncoder, pipelines[172], layouts[172], infinityBuf, [buf_7, buf_146], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[173], layouts[173], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[174], layouts[174], infinityBuf, [buf_7, buf_146, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[175], layouts[175], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[176], layouts[176], infinityBuf, [buf_101, buf_146, buf_8, buf_9, buf_149, buf_150], [4096, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[177], layouts[177], infinityBuf, [buf_146, buf_137, buf_101, buf_151, buf_152], [8, 64, 32]);
        addComputePass(device, commandEncoder, pipelines[178], layouts[178], infinityBuf, [buf_7, buf_146], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[179], layouts[179], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[180], layouts[180], infinityBuf, [buf_7, buf_146, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[181], layouts[181], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[182], layouts[182], infinityBuf, [buf_137, buf_146, buf_8, buf_9, buf_153, buf_154], [4096, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[183], layouts[183], infinityBuf, [buf_101, buf_137, buf_155, buf_156], [8, 64, 32]);
        addComputePass(device, commandEncoder, pipelines[184], layouts[184], infinityBuf, [buf_7, buf_101], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[185], layouts[185], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[186], layouts[186], infinityBuf, [buf_7, buf_101, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[187], layouts[187], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[188], layouts[188], infinityBuf, [buf_137, buf_101, buf_8, buf_9, buf_157, buf_158], [4096, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[189], layouts[189], infinityBuf, [buf_159, buf_146, buf_137, buf_160, buf_161], [8, 64, 32]);
        addComputePass(device, commandEncoder, pipelines[190], layouts[190], infinityBuf, [buf_7, buf_159], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[191], layouts[191], infinityBuf, [buf_8, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[192], layouts[192], infinityBuf, [buf_7, buf_159, buf_8], [4, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[193], layouts[193], infinityBuf, [buf_9, buf_7], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[194], layouts[194], infinityBuf, [buf_101, buf_159, buf_8, buf_9, buf_162, buf_163], [4096, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[195], layouts[195], infinityBuf, [buf_164, buf_101, buf_165, buf_166], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[196], layouts[196], infinityBuf, [output0, buf_164], [2048, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Uint8Array(gpuReadBuffer0.size/1);
        resultBuffer0.set(new Uint8Array(gpuReadBuffer0.getMappedRange()));
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
export default decoder;
