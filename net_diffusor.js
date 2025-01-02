
const diffusor = (() => {
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

const E_n62 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data1[0];
  data0[0] = sqrt(val0);
}`;

const E_n63 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data1[0];
  data0[0] = sqrt((1.0f-val0));
}`;

const E_128_32_4n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
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
  data0[alu1] = (f16(val1));
  data0[alu2] = (f16(val2));
  data0[alu3] = (f16(val3));
  data0[alu0] = (f16(val0));
}`;

const r2_160_16_10 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data1[0];
  var alu0 = (gidx0+(lidx0*10)+-149);
  var alu1 = -select(alu0,0,(alu0<0));
  temp0[lidx0] = -select(alu1,-10,(alu1<-10));
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0;
    for (var ridx0 = 0; ridx0 < 16; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    var alu7 = (exp2(((f32((acc0+-1)))*-0.08304820360969871f))*val0);
    data2[gidx0] = sin(alu7);
    data0[gidx0] = sin((1.5707963267948966f-alu7));
  }
}`;

const E_1848_32_2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1848 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+(gidx0<<5));
  var val0 = data1[alu0];
  var val1 = data2[alu0];
  data0[alu0] = (f16(val0));
  data0[(alu0+59136)] = (f16(val1));
}`;

const r_2_80_8_8_16_4_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
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
    var val37 = select((f16(0.0f)), data1[(alu8+-65)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-64)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-63)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-62)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-61)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-60)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+63)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+64)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+65)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+66)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+67)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data3[(alu26+1)];
  var val56 = data3[(alu26+2)];
  var val57 = data3[(alu26+3)];
  var alu27 = ((gidx1<<14)+(gidx2*1310720)+alu0+alu1+alu2);
  data0[alu27] = ((f16(acc0))+val54);
  data0[(alu27+1)] = ((f16(acc4))+val54);
  data0[(alu27+2)] = ((f16(acc8))+val54);
  data0[(alu27+3)] = ((f16(acc12))+val54);
  data0[(alu27+4096)] = ((f16(acc1))+val55);
  data0[(alu27+4097)] = ((f16(acc5))+val55);
  data0[(alu27+4098)] = ((f16(acc9))+val55);
  data0[(alu27+4099)] = ((f16(acc13))+val55);
  data0[(alu27+8192)] = ((f16(acc2))+val56);
  data0[(alu27+8193)] = ((f16(acc6))+val56);
  data0[(alu27+8194)] = ((f16(acc10))+val56);
  data0[(alu27+8195)] = ((f16(acc14))+val56);
  data0[(alu27+12288)] = ((f16(acc3))+val57);
  data0[(alu27+12289)] = ((f16(acc7))+val57);
  data0[(alu27+12290)] = ((f16(acc11))+val57);
  data0[(alu27+12291)] = ((f16(acc15))+val57);
}`;

const E_5_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = ((lidx0+(gidx0<<4))<40);
  var alu1 = (alu0!=true);
  var alu2 = ((gidx0<<6)+(lidx0<<2));
  var val0 = select(0.0f, data1[alu2], alu0);
  var val1 = select(0.0f, data2[(alu2+-160)], alu1);
  var val2 = select(0.0f, data2[(alu2+-159)], alu1);
  var val3 = select(0.0f, data2[(alu2+-158)], alu1);
  var val4 = select(0.0f, data2[(alu2+-157)], alu1);
  var alu3 = (alu2+1);
  var val5 = select(0.0f, data1[alu3], alu0);
  var alu4 = (alu2+2);
  var val6 = select(0.0f, data1[alu4], alu0);
  var alu5 = (alu2+3);
  var val7 = select(0.0f, data1[alu5], alu0);
  data0[alu3] = (f16((val5+val2)));
  data0[alu4] = (f16((val6+val3)));
  data0[alu5] = (f16((val7+val4)));
  data0[alu2] = (f16((val0+val1)));
}`;

const r_77_5_2_16_192_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 192; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*49152)+(lidx1*3072)+alu0);
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
    var val12 = data2[(alu1+2304)];
    var val13 = data2[(alu1+2305)];
    var val14 = data2[(alu1+2306)];
    var val15 = data2[(alu1+2307)];
    var alu2 = ((gidx1*1536)+(lidx0*768)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
  }
  var alu8 = ((gidx0<<6)+(gidx1*640)+(lidx0*320)+(lidx1<<2));
  data0[alu8] = (f16(acc0));
  data0[(alu8+1)] = (f16(acc1));
  data0[(alu8+2)] = (f16(acc2));
  data0[(alu8+3)] = (f16(acc3));
}`;

const r_77_10_2_16_192_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 192; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*49152)+(lidx1*3072)+alu0);
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
    var val12 = data2[(alu1+2304)];
    var val13 = data2[(alu1+2305)];
    var val14 = data2[(alu1+2306)];
    var val15 = data2[(alu1+2307)];
    var alu2 = ((gidx1*1536)+(lidx0*768)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
  }
  var alu8 = ((gidx0<<6)+(gidx1*1280)+(lidx0*640)+(lidx1<<2));
  data0[alu8] = (f16(acc0));
  data0[(alu8+1)] = (f16(acc1));
  data0[(alu8+2)] = (f16(acc2));
  data0[(alu8+3)] = (f16(acc3));
}`;

const r_77_20_2_16_192_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 77 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 192; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*49152)+(lidx1*3072)+alu0);
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
    var val12 = data2[(alu1+2304)];
    var val13 = data2[(alu1+2305)];
    var val14 = data2[(alu1+2306)];
    var val15 = data2[(alu1+2307)];
    var alu2 = ((gidx1*1536)+(lidx0*768)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
  }
  var alu8 = ((gidx0<<6)+(gidx1*2560)+(lidx0*1280)+(lidx1<<2));
  data0[alu8] = (f16(acc0));
  data0[(alu8+1)] = (f16(acc1));
  data0[(alu8+2)] = (f16(acc2));
  data0[(alu8+3)] = (f16(acc3));
}`;

const r_512_32_40_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var alu0 = ((gidx0*5120)+(lidx0*160)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+(f32(val3))+(f32(val2))+(f32(val1))+(f32(val0)));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_1280_16_20 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1280 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data3[gidx0];
  var alu0 = (lidx0*20);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 20; ridx0++) {
    var val1 = data2[((gidx0*320)+alu0+ridx0)];
    var val2 = data1[(alu0+ridx0)];
    acc0 = (acc0+(f32((val2*val1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val3 = temp0[ridx1];
      acc1 = (acc1+val3);
    }
    var cast0 = (f16(acc1));
    data0[gidx0] = ((1/(exp2(((-cast0-val0)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(cast0+val0));
  }
}`;

const r_64_16_16 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = (f16((acc1*2.441406286379788e-05f)));
  }
}`;

const r_1280_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1280 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data3[gidx0];
  var alu0 = (lidx0*80);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data2[((gidx0*1280)+alu0+ridx0)];
    var val2 = data1[(alu0+ridx0)];
    acc0 = (acc0+(f32((val2*val1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val3 = temp0[ridx1];
      acc1 = (acc1+val3);
    }
    var cast0 = (f16(acc1));
    data0[gidx0] = ((1/(exp2(((-cast0-val0)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(cast0+val0));
  }
}`;

const r_8_4_8_16_40_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var alu0 = ((gidx0*10240)+(gidx1*327680)+(lidx0*40960)+(lidx1*640)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+160)];
    var val6 = data1[(alu0+161)];
    var val7 = data1[(alu0+162)];
    var val8 = data1[(alu0+163)];
    var val9 = data1[(alu0+320)];
    var val10 = data1[(alu0+321)];
    var val11 = data1[(alu0+322)];
    var val12 = data1[(alu0+323)];
    var val13 = data1[(alu0+480)];
    var val14 = data1[(alu0+481)];
    var val15 = data1[(alu0+482)];
    var val16 = data1[(alu0+483)];
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
    acc0 = (acc0+(f32((alu3*alu3)))+(f32((alu2*alu2)))+(f32((alu1*alu1)))+(f32((alu16*alu16))));
    acc1 = (acc1+(f32((alu7*alu7)))+(f32((alu6*alu6)))+(f32((alu4*alu4)))+(f32((alu5*alu5))));
    acc2 = (acc2+(f32((alu11*alu11)))+(f32((alu10*alu10)))+(f32((alu8*alu8)))+(f32((alu9*alu9))));
    acc3 = (acc3+(f32((alu15*alu15)))+(f32((alu14*alu14)))+(f32((alu12*alu12)))+(f32((alu13*alu13))));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_320_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 320 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data3[gidx0];
  var alu0 = (lidx0*80);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data2[((gidx0*1280)+alu0+ridx0)];
    var val2 = data1[(alu0+ridx0)];
    acc0 = (acc0+(f32((val2*val1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val3 = temp0[ridx1];
      acc1 = (acc1+val3);
    }
    data0[gidx0] = ((f16(acc1))+val0);
  }
}`;

const r_640_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 640 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data3[gidx0];
  var alu0 = (lidx0*80);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data2[((gidx0*1280)+alu0+ridx0)];
    var val2 = data1[(alu0+ridx0)];
    acc0 = (acc0+(f32((val2*val1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val3 = temp0[ridx1];
      acc1 = (acc1+val3);
    }
    data0[gidx0] = ((f16(acc1))+val0);
  }
}`;

const r_1280_16_80n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1280 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data3[gidx0];
  var alu0 = (lidx0*80);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data2[((gidx0*1280)+alu0+ridx0)];
    var val2 = data1[(alu0+ridx0)];
    acc0 = (acc0+(f32((val2*val1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val3 = temp0[ridx1];
      acc1 = (acc1+val3);
    }
    data0[gidx0] = ((f16(acc1))+val0);
  }
}`;

const r_64_16_16n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = sqrt((1/((f16((acc1*2.441406286379788e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_40_64_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/10));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<15)+(gidx2*1310720)+(gidx0<<6)+(lidx0<<12)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_80_8_8_16_320_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
  var alu1 = (gidx0<<9);
  var alu2 = (lidx0<<6);
  var alu3 = (lidx1<<2);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (((gidx0+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx0<<3))<63);
  var alu7 = (lidx1<15);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu8 = ((gidx1*11520)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+2880)];
    var val10 = data2[(alu8+2881)];
    var val11 = data2[(alu8+2882)];
    var val12 = data2[(alu8+2883)];
    var val13 = data2[(alu8+2884)];
    var val14 = data2[(alu8+2885)];
    var val15 = data2[(alu8+2886)];
    var val16 = data2[(alu8+2887)];
    var val17 = data2[(alu8+2888)];
    var val18 = data2[(alu8+5760)];
    var val19 = data2[(alu8+5761)];
    var val20 = data2[(alu8+5762)];
    var val21 = data2[(alu8+5763)];
    var val22 = data2[(alu8+5764)];
    var val23 = data2[(alu8+5765)];
    var val24 = data2[(alu8+5766)];
    var val25 = data2[(alu8+5767)];
    var val26 = data2[(alu8+5768)];
    var val27 = data2[(alu8+8640)];
    var val28 = data2[(alu8+8641)];
    var val29 = data2[(alu8+8642)];
    var val30 = data2[(alu8+8643)];
    var val31 = data2[(alu8+8644)];
    var val32 = data2[(alu8+8645)];
    var val33 = data2[(alu8+8646)];
    var val34 = data2[(alu8+8647)];
    var val35 = data2[(alu8+8648)];
    var alu9 = (alu1+alu2+alu0+(ridx0<<12)+alu3);
    var val36 = data1[alu9];
    var val37 = select((f16(0.0f)), data1[(alu9+-65)], (alu4&alu5));
    var val38 = select((f16(0.0f)), data1[(alu9+-64)], alu5);
    var val39 = select((f16(0.0f)), data1[(alu9+-63)], alu5);
    var val40 = select((f16(0.0f)), data1[(alu9+-62)], alu5);
    var val41 = select((f16(0.0f)), data1[(alu9+-61)], alu5);
    var val42 = select((f16(0.0f)), data1[(alu9+-60)], (alu7&alu5));
    var val43 = select((f16(0.0f)), data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select((f16(0.0f)), data1[(alu9+4)], alu7);
    var val48 = select((f16(0.0f)), data1[(alu9+63)], (alu6&alu4));
    var val49 = select((f16(0.0f)), data1[(alu9+64)], alu6);
    var val50 = select((f16(0.0f)), data1[(alu9+65)], alu6);
    var val51 = select((f16(0.0f)), data1[(alu9+66)], alu6);
    var val52 = select((f16(0.0f)), data1[(alu9+67)], alu6);
    var val53 = select((f16(0.0f)), data1[(alu9+68)], (alu7&alu6));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu27 = (gidx1<<2);
  var val54 = data3[alu27];
  var val55 = data4[alu27];
  var alu28 = (alu27+1);
  var val56 = data3[alu28];
  var val57 = data4[alu28];
  var alu29 = (alu27+2);
  var val58 = data3[alu29];
  var val59 = data4[alu29];
  var alu30 = (alu27+3);
  var val60 = data3[alu30];
  var val61 = data4[alu30];
  var alu31 = ((gidx1<<14)+alu0+alu1+alu2+alu3);
  data0[alu31] = (val55+(f16(acc0))+val54);
  data0[(alu31+1)] = (val55+(f16(acc4))+val54);
  data0[(alu31+2)] = (val55+(f16(acc8))+val54);
  data0[(alu31+3)] = (val55+(f16(acc12))+val54);
  data0[(alu31+4096)] = (val57+(f16(acc1))+val56);
  data0[(alu31+4097)] = (val57+(f16(acc5))+val56);
  data0[(alu31+4098)] = (val57+(f16(acc9))+val56);
  data0[(alu31+4099)] = (val57+(f16(acc13))+val56);
  data0[(alu31+8192)] = (val59+(f16(acc2))+val58);
  data0[(alu31+8193)] = (val59+(f16(acc6))+val58);
  data0[(alu31+8194)] = (val59+(f16(acc10))+val58);
  data0[(alu31+8195)] = (val59+(f16(acc14))+val58);
  data0[(alu31+12288)] = (val61+(f16(acc3))+val60);
  data0[(alu31+12289)] = (val61+(f16(acc7))+val60);
  data0[(alu31+12290)] = (val61+(f16(acc11))+val60);
  data0[(alu31+12291)] = (val61+(f16(acc15))+val60);
}`;

const r_2_80_8_8_16_320_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
  var alu1 = (gidx0<<9);
  var alu2 = (lidx0<<6);
  var alu3 = (lidx1<<2);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (((gidx0+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx0<<3))<63);
  var alu7 = (lidx1<15);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu8 = ((gidx1*11520)+(ridx0*9));
    var val0 = data3[alu8];
    var val1 = data3[(alu8+1)];
    var val2 = data3[(alu8+2)];
    var val3 = data3[(alu8+3)];
    var val4 = data3[(alu8+4)];
    var val5 = data3[(alu8+5)];
    var val6 = data3[(alu8+6)];
    var val7 = data3[(alu8+7)];
    var val8 = data3[(alu8+8)];
    var val9 = data3[(alu8+2880)];
    var val10 = data3[(alu8+2881)];
    var val11 = data3[(alu8+2882)];
    var val12 = data3[(alu8+2883)];
    var val13 = data3[(alu8+2884)];
    var val14 = data3[(alu8+2885)];
    var val15 = data3[(alu8+2886)];
    var val16 = data3[(alu8+2887)];
    var val17 = data3[(alu8+2888)];
    var val18 = data3[(alu8+5760)];
    var val19 = data3[(alu8+5761)];
    var val20 = data3[(alu8+5762)];
    var val21 = data3[(alu8+5763)];
    var val22 = data3[(alu8+5764)];
    var val23 = data3[(alu8+5765)];
    var val24 = data3[(alu8+5766)];
    var val25 = data3[(alu8+5767)];
    var val26 = data3[(alu8+5768)];
    var val27 = data3[(alu8+8640)];
    var val28 = data3[(alu8+8641)];
    var val29 = data3[(alu8+8642)];
    var val30 = data3[(alu8+8643)];
    var val31 = data3[(alu8+8644)];
    var val32 = data3[(alu8+8645)];
    var val33 = data3[(alu8+8646)];
    var val34 = data3[(alu8+8647)];
    var val35 = data3[(alu8+8648)];
    var alu9 = (alu1+alu2+alu0+(ridx0<<12)+alu3);
    var val36 = data2[alu9];
    var val37 = select((f16(0.0f)), data2[(alu9+-65)], (alu4&alu5));
    var val38 = select((f16(0.0f)), data2[(alu9+-64)], alu5);
    var val39 = select((f16(0.0f)), data2[(alu9+-63)], alu5);
    var val40 = select((f16(0.0f)), data2[(alu9+-62)], alu5);
    var val41 = select((f16(0.0f)), data2[(alu9+-61)], alu5);
    var val42 = select((f16(0.0f)), data2[(alu9+-60)], (alu7&alu5));
    var val43 = select((f16(0.0f)), data2[(alu9+-1)], alu4);
    var val44 = data2[(alu9+1)];
    var val45 = data2[(alu9+2)];
    var val46 = data2[(alu9+3)];
    var val47 = select((f16(0.0f)), data2[(alu9+4)], alu7);
    var val48 = select((f16(0.0f)), data2[(alu9+63)], (alu6&alu4));
    var val49 = select((f16(0.0f)), data2[(alu9+64)], alu6);
    var val50 = select((f16(0.0f)), data2[(alu9+65)], alu6);
    var val51 = select((f16(0.0f)), data2[(alu9+66)], alu6);
    var val52 = select((f16(0.0f)), data2[(alu9+67)], alu6);
    var val53 = select((f16(0.0f)), data2[(alu9+68)], (alu7&alu6));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu27 = (gidx1<<2);
  var val54 = data4[alu27];
  var val55 = data4[(alu27+1)];
  var val56 = data4[(alu27+2)];
  var val57 = data4[(alu27+3)];
  var alu28 = ((gidx1<<14)+alu0+alu1+alu2+alu3);
  var val58 = data1[alu28];
  var alu29 = (alu28+1);
  var val59 = data1[alu29];
  var alu30 = (alu28+2);
  var val60 = data1[alu30];
  var alu31 = (alu28+3);
  var val61 = data1[alu31];
  var alu32 = (alu28+4096);
  var val62 = data1[alu32];
  var alu33 = (alu28+4097);
  var val63 = data1[alu33];
  var alu34 = (alu28+4098);
  var val64 = data1[alu34];
  var alu35 = (alu28+4099);
  var val65 = data1[alu35];
  var alu36 = (alu28+8192);
  var val66 = data1[alu36];
  var alu37 = (alu28+8193);
  var val67 = data1[alu37];
  var alu38 = (alu28+8194);
  var val68 = data1[alu38];
  var alu39 = (alu28+8195);
  var val69 = data1[alu39];
  var alu40 = (alu28+12288);
  var val70 = data1[alu40];
  var alu41 = (alu28+12289);
  var val71 = data1[alu41];
  var alu42 = (alu28+12290);
  var val72 = data1[alu42];
  var alu43 = (alu28+12291);
  var val73 = data1[alu43];
  data0[alu29] = (val59+(f16(acc4))+val54);
  data0[alu30] = (val60+(f16(acc8))+val54);
  data0[alu31] = (val61+(f16(acc12))+val54);
  data0[alu32] = (val62+(f16(acc1))+val55);
  data0[alu33] = (val63+(f16(acc5))+val55);
  data0[alu34] = (val64+(f16(acc9))+val55);
  data0[alu35] = (val65+(f16(acc13))+val55);
  data0[alu36] = (val66+(f16(acc2))+val56);
  data0[alu37] = (val67+(f16(acc6))+val56);
  data0[alu38] = (val68+(f16(acc10))+val56);
  data0[alu39] = (val69+(f16(acc14))+val56);
  data0[alu40] = (val70+(f16(acc3))+val57);
  data0[alu41] = (val71+(f16(acc7))+val57);
  data0[alu42] = (val72+(f16(acc11))+val57);
  data0[alu43] = (val73+(f16(acc15))+val57);
  data0[alu28] = (val58+(f16(acc0))+val54);
}`;

const E_2_40_64_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/10));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<15)+(gidx2*1310720)+(gidx0<<6)+(lidx0<<12)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  data0[alu3] = (val1+(val0*val3*(val5-val2)));
  data0[alu4] = (val1+(val0*val3*(val6-val2)));
  data0[alu5] = (val1+(val0*val3*(val7-val2)));
  data0[alu2] = (val1+(val0*val3*(val4-val2)));
}`;

const r_2_10_64_8_16_80_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 10 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx1<<2);
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu3 = ((gidx1*10240)+(lidx0*1280)+(ridx0<<2));
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+320)];
    var val5 = data2[(alu3+321)];
    var val6 = data2[(alu3+322)];
    var val7 = data2[(alu3+323)];
    var val8 = data2[(alu3+640)];
    var val9 = data2[(alu3+641)];
    var val10 = data2[(alu3+642)];
    var val11 = data2[(alu3+643)];
    var val12 = data2[(alu3+960)];
    var val13 = data2[(alu3+961)];
    var val14 = data2[(alu3+962)];
    var val15 = data2[(alu3+963)];
    var alu4 = (alu1+alu0+alu2+(ridx0<<14));
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
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
  }
  var alu22 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = ((gidx1<<17)+alu0+alu1+(lidx0<<14)+alu2);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val32);
  data0[(alu23+2)] = ((f16(acc8))+val32);
  data0[(alu23+3)] = ((f16(acc12))+val32);
  data0[(alu23+4096)] = ((f16(acc1))+val33);
  data0[(alu23+4097)] = ((f16(acc5))+val33);
  data0[(alu23+4098)] = ((f16(acc9))+val33);
  data0[(alu23+4099)] = ((f16(acc13))+val33);
  data0[(alu23+8192)] = ((f16(acc2))+val34);
  data0[(alu23+8193)] = ((f16(acc6))+val34);
  data0[(alu23+8194)] = ((f16(acc10))+val34);
  data0[(alu23+8195)] = ((f16(acc14))+val34);
  data0[(alu23+12288)] = ((f16(acc3))+val35);
  data0[(alu23+12289)] = ((f16(acc7))+val35);
  data0[(alu23+12290)] = ((f16(acc11))+val35);
  data0[(alu23+12291)] = ((f16(acc15))+val35);
}`;

const r_256_2_16_80_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<4);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu1 = (lidx1+alu0+(lidx0*1310720)+(ridx0<<14));
    var val0 = data1[alu1];
    var val1 = data1[(alu1+4096)];
    var val2 = data1[(alu1+8192)];
    var val3 = data1[(alu1+12288)];
    acc0 = (acc0+(f32(val3))+(f32(val2))+(f32(val0))+(f32(val1)));
  }
  data0[(lidx1+alu0+(lidx0<<12))] = (f16((acc0*0.0031250000465661287f)));
}`;

const r_64_2_16_80_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = (alu0+(lidx0<<12)+alu1);
  var alu3 = (alu2+1);
  var alu4 = (alu2+2);
  var alu5 = (alu2+3);
  var val0 = data2[alu3];
  var val1 = data2[alu4];
  var val2 = data2[alu5];
  var val3 = data2[alu2];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu6 = (alu0+(lidx0*1310720)+alu1+(ridx0<<14));
    var val4 = data1[alu6];
    var val5 = data1[(alu6+1)];
    var val6 = data1[(alu6+2)];
    var val7 = data1[(alu6+3)];
    var val8 = data1[(alu6+4096)];
    var val9 = data1[(alu6+4097)];
    var val10 = data1[(alu6+4098)];
    var val11 = data1[(alu6+4099)];
    var val12 = data1[(alu6+8192)];
    var val13 = data1[(alu6+8193)];
    var val14 = data1[(alu6+8194)];
    var val15 = data1[(alu6+8195)];
    var val16 = data1[(alu6+12288)];
    var val17 = data1[(alu6+12289)];
    var val18 = data1[(alu6+12290)];
    var val19 = data1[(alu6+12291)];
    var alu7 = (val5-val0);
    var alu8 = (val6-val1);
    var alu9 = (val7-val2);
    var alu10 = (val8-val3);
    var alu11 = (val9-val0);
    var alu12 = (val10-val1);
    var alu13 = (val11-val2);
    var alu14 = (val12-val3);
    var alu15 = (val13-val0);
    var alu16 = (val14-val1);
    var alu17 = (val15-val2);
    var alu18 = (val16-val3);
    var alu19 = (val17-val0);
    var alu20 = (val18-val1);
    var alu21 = (val19-val2);
    var alu22 = (val4-val3);
    acc0 = (acc0+(f32((alu18*alu18)))+(f32((alu14*alu14)))+(f32((alu10*alu10)))+(f32((alu22*alu22))));
    acc1 = (acc1+(f32((alu19*alu19)))+(f32((alu15*alu15)))+(f32((alu7*alu7)))+(f32((alu11*alu11))));
    acc2 = (acc2+(f32((alu20*alu20)))+(f32((alu16*alu16)))+(f32((alu8*alu8)))+(f32((alu12*alu12))));
    acc3 = (acc3+(f32((alu21*alu21)))+(f32((alu17*alu17)))+(f32((alu9*alu9)))+(f32((alu13*alu13))));
  }
  data0[alu2] = sqrt((1/((f16((acc0*0.0031250000465661287f)))+(f16(1e-05f)))));
  data0[alu3] = sqrt((1/((f16((acc1*0.0031250000465661287f)))+(f16(1e-05f)))));
  data0[alu4] = sqrt((1/((f16((acc2*0.0031250000465661287f)))+(f16(1e-05f)))));
  data0[alu5] = sqrt((1/((f16((acc3*0.0031250000465661287f)))+(f16(1e-05f)))));
}`;

const E_2_128_5_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 128 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
  var alu1 = (gidx0<<6);
  var alu2 = (gidx1<<5);
  var alu3 = (lidx0<<2);
  var alu4 = (alu2+(gidx2<<12)+alu3);
  var val0 = data2[alu4];
  var val1 = data3[alu4];
  var alu5 = (alu4+1);
  var val2 = data2[alu5];
  var val3 = data3[alu5];
  var alu6 = (alu4+2);
  var val4 = data2[alu6];
  var val5 = data3[alu6];
  var alu7 = (alu4+3);
  var val6 = data2[alu7];
  var val7 = data3[alu7];
  var alu8 = (lidx1<<2);
  var alu9 = (alu1+alu8);
  var val8 = data4[alu9];
  var val9 = data5[alu9];
  var alu10 = (alu9+1);
  var val10 = data4[alu10];
  var val11 = data5[alu10];
  var alu11 = (alu9+2);
  var val12 = data4[alu11];
  var val13 = data5[alu11];
  var alu12 = (alu9+3);
  var val14 = data4[alu12];
  var val15 = data5[alu12];
  var alu13 = (alu2+alu0+(gidx0<<18)+alu3+(lidx1<<14));
  var val16 = data1[alu13];
  var val17 = data1[(alu13+1)];
  var val18 = data1[(alu13+2)];
  var val19 = data1[(alu13+3)];
  var val20 = data1[(alu13+4096)];
  var val21 = data1[(alu13+4097)];
  var val22 = data1[(alu13+4098)];
  var val23 = data1[(alu13+4099)];
  var val24 = data1[(alu13+8192)];
  var val25 = data1[(alu13+8193)];
  var val26 = data1[(alu13+8194)];
  var val27 = data1[(alu13+8195)];
  var val28 = data1[(alu13+12288)];
  var val29 = data1[(alu13+12289)];
  var val30 = data1[(alu13+12290)];
  var val31 = data1[(alu13+12291)];
  var alu14 = ((gidx1*10240)+alu0+alu1+(lidx0*1280)+alu8);
  data0[(alu14+320)] = (val9+(val8*val3*(val17-val2)));
  data0[(alu14+640)] = (val9+(val8*val5*(val18-val4)));
  data0[(alu14+960)] = (val9+(val8*val7*(val19-val6)));
  data0[(alu14+1)] = (val11+(val10*val1*(val20-val0)));
  data0[(alu14+321)] = (val11+(val10*val3*(val21-val2)));
  data0[(alu14+641)] = (val11+(val10*val5*(val22-val4)));
  data0[(alu14+961)] = (val11+(val10*val7*(val23-val6)));
  data0[(alu14+2)] = (val13+(val12*val1*(val24-val0)));
  data0[(alu14+322)] = (val13+(val12*val3*(val25-val2)));
  data0[(alu14+642)] = (val13+(val12*val5*(val26-val4)));
  data0[(alu14+962)] = (val13+(val12*val7*(val27-val6)));
  data0[(alu14+3)] = (val15+(val14*val1*(val28-val0)));
  data0[(alu14+323)] = (val15+(val14*val3*(val29-val2)));
  data0[(alu14+643)] = (val15+(val14*val5*(val30-val4)));
  data0[(alu14+963)] = (val15+(val14*val7*(val31-val6)));
  data0[alu14] = (val9+(val8*val1*(val16-val0)));
}`;

const r_256_5_8_16_80_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 256 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1*10240);
  var alu1 = (lidx0*1280);
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu2 = (ridx0<<2);
    var alu3 = ((gidx0*20480)+(lidx1*1280)+alu2);
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+320)];
    var val5 = data2[(alu3+321)];
    var val6 = data2[(alu3+322)];
    var val7 = data2[(alu3+323)];
    var val8 = data2[(alu3+640)];
    var val9 = data2[(alu3+641)];
    var val10 = data2[(alu3+642)];
    var val11 = data2[(alu3+643)];
    var val12 = data2[(alu3+960)];
    var val13 = data2[(alu3+961)];
    var val14 = data2[(alu3+962)];
    var val15 = data2[(alu3+963)];
    var alu4 = (alu0+alu1+alu2);
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+320)];
    var val21 = data1[(alu4+321)];
    var val22 = data1[(alu4+322)];
    var val23 = data1[(alu4+323)];
    var val24 = data1[(alu4+640)];
    var val25 = data1[(alu4+641)];
    var val26 = data1[(alu4+642)];
    var val27 = data1[(alu4+643)];
    var val28 = data1[(alu4+960)];
    var val29 = data1[(alu4+961)];
    var val30 = data1[(alu4+962)];
    var val31 = data1[(alu4+963)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu22 = ((gidx0<<6)+alu0+alu1+(lidx1<<2));
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+320)] = (f16(acc1));
  data0[(alu22+321)] = (f16(acc5));
  data0[(alu22+322)] = (f16(acc9));
  data0[(alu22+323)] = (f16(acc13));
  data0[(alu22+640)] = (f16(acc2));
  data0[(alu22+641)] = (f16(acc6));
  data0[(alu22+642)] = (f16(acc10));
  data0[(alu22+643)] = (f16(acc14));
  data0[(alu22+960)] = (f16(acc3));
  data0[(alu22+961)] = (f16(acc7));
  data0[(alu22+962)] = (f16(acc11));
  data0[(alu22+963)] = (f16(acc15));
}`;

const r_2_8_128_64_8_16_10_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8192 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0>>6);
  var alu1 = (gidx0&63);
  var alu2 = ((gidx1*40)+(gidx2*1310720));
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
  for (var ridx0 = 0; ridx0 < 10; ridx0++) {
    var alu3 = (ridx0<<2);
    var alu4 = (alu2+(alu0*10240)+(lidx0*1280)+alu3);
    var val0 = data1[alu4];
    var val1 = data1[(alu4+1)];
    var val2 = data1[(alu4+2)];
    var val3 = data1[(alu4+3)];
    var val4 = data1[(alu4+320)];
    var val5 = data1[(alu4+321)];
    var val6 = data1[(alu4+322)];
    var val7 = data1[(alu4+323)];
    var val8 = data1[(alu4+640)];
    var val9 = data1[(alu4+641)];
    var val10 = data1[(alu4+642)];
    var val11 = data1[(alu4+643)];
    var val12 = data1[(alu4+960)];
    var val13 = data1[(alu4+961)];
    var val14 = data1[(alu4+962)];
    var val15 = data1[(alu4+963)];
    var alu5 = (alu2+(alu1*20480)+(lidx1*1280)+alu3);
    var val16 = data2[alu5];
    var val17 = data2[(alu5+1)];
    var val18 = data2[(alu5+2)];
    var val19 = data2[(alu5+3)];
    var val20 = data2[(alu5+320)];
    var val21 = data2[(alu5+321)];
    var val22 = data2[(alu5+322)];
    var val23 = data2[(alu5+323)];
    var val24 = data2[(alu5+640)];
    var val25 = data2[(alu5+641)];
    var val26 = data2[(alu5+642)];
    var val27 = data2[(alu5+643)];
    var val28 = data2[(alu5+960)];
    var val29 = data2[(alu5+961)];
    var val30 = data2[(alu5+962)];
    var val31 = data2[(alu5+963)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc2 = (acc2+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc3 = (acc3+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc4 = (acc4+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc7 = (acc7+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc8 = (acc8+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc9 = (acc9+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc12 = (acc12+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc13 = (acc13+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc14 = (acc14+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu23 = ((gidx1<<24)+(gidx2<<27)+(alu0<<17)+(alu1<<6)+(lidx0<<14)+(lidx1<<2));
  data0[alu23] = (acc0*0.15811388194561005f);
  data0[(alu23+1)] = (acc4*0.15811388194561005f);
  data0[(alu23+2)] = (acc8*0.15811388194561005f);
  data0[(alu23+3)] = (acc12*0.15811388194561005f);
  data0[(alu23+4096)] = (acc1*0.15811388194561005f);
  data0[(alu23+4097)] = (acc5*0.15811388194561005f);
  data0[(alu23+4098)] = (acc9*0.15811388194561005f);
  data0[(alu23+4099)] = (acc13*0.15811388194561005f);
  data0[(alu23+8192)] = (acc2*0.15811388194561005f);
  data0[(alu23+8193)] = (acc6*0.15811388194561005f);
  data0[(alu23+8194)] = (acc10*0.15811388194561005f);
  data0[(alu23+8195)] = (acc14*0.15811388194561005f);
  data0[(alu23+12288)] = (acc3*0.15811388194561005f);
  data0[(alu23+12289)] = (acc7*0.15811388194561005f);
  data0[(alu23+12290)] = (acc11*0.15811388194561005f);
  data0[(alu23+12291)] = (acc15*0.15811388194561005f);
}`;

const r_2048_32_1024_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
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

const r_512_32_1024_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
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

const E_8192_64_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 8192 */
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
  data0[alu2] = (f16((exp2(((val3-val0)*1.4426950408889634f))*alu5)));
  data0[alu3] = (f16((exp2(((val4-val0)*1.4426950408889634f))*alu5)));
  data0[alu4] = (f16((exp2(((val5-val0)*1.4426950408889634f))*alu5)));
  data0[alu1] = (f16((exp2(((val2-val0)*1.4426950408889634f))*alu5)));
}`;

const r_2_2_64_5_4_16_2_1024_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(4,16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 320 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (gidx2*1310720);
  var alu1 = (gidx0/5);
  var alu2 = ((gidx0%5)<<3);
  var alu3 = (lidx2<<2);
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
    var alu4 = ((gidx1*160)+alu0+alu2+(lidx0*40)+alu3+(ridx0*1280));
    var val0 = data2[alu4];
    var val1 = data2[(alu4+1)];
    var val2 = data2[(alu4+2)];
    var val3 = data2[(alu4+3)];
    var val4 = data2[(alu4+320)];
    var val5 = data2[(alu4+321)];
    var val6 = data2[(alu4+322)];
    var val7 = data2[(alu4+323)];
    var val8 = data2[(alu4+640)];
    var val9 = data2[(alu4+641)];
    var val10 = data2[(alu4+642)];
    var val11 = data2[(alu4+643)];
    var val12 = data2[(alu4+960)];
    var val13 = data2[(alu4+961)];
    var val14 = data2[(alu4+962)];
    var val15 = data2[(alu4+963)];
    var alu5 = ((gidx1<<26)+(gidx2<<27)+(alu1<<18)+(lidx0<<24)+(lidx1<<14)+(ridx0<<2));
    var val16 = data1[alu5];
    var val17 = data1[(alu5+1)];
    var val18 = data1[(alu5+2)];
    var val19 = data1[(alu5+3)];
    var val20 = data1[(alu5+4096)];
    var val21 = data1[(alu5+4097)];
    var val22 = data1[(alu5+4098)];
    var val23 = data1[(alu5+4099)];
    var val24 = data1[(alu5+8192)];
    var val25 = data1[(alu5+8193)];
    var val26 = data1[(alu5+8194)];
    var val27 = data1[(alu5+8195)];
    var val28 = data1[(alu5+12288)];
    var val29 = data1[(alu5+12289)];
    var val30 = data1[(alu5+12290)];
    var val31 = data1[(alu5+12291)];
    acc0 = (acc0+(f32((val12*val19)))+(f32((val8*val18)))+(f32((val4*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val12*val23)))+(f32((val8*val22)))+(f32((val4*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val12*val27)))+(f32((val8*val26)))+(f32((val4*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val12*val31)))+(f32((val8*val30)))+(f32((val4*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val13*val19)))+(f32((val9*val18)))+(f32((val1*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val13*val23)))+(f32((val9*val22)))+(f32((val1*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val13*val27)))+(f32((val9*val26)))+(f32((val1*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val13*val31)))+(f32((val9*val30)))+(f32((val1*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val14*val19)))+(f32((val10*val18)))+(f32((val2*val16)))+(f32((val6*val17))));
    acc9 = (acc9+(f32((val14*val23)))+(f32((val10*val22)))+(f32((val2*val20)))+(f32((val6*val21))));
    acc10 = (acc10+(f32((val14*val27)))+(f32((val10*val26)))+(f32((val2*val24)))+(f32((val6*val25))));
    acc11 = (acc11+(f32((val14*val31)))+(f32((val10*val30)))+(f32((val2*val28)))+(f32((val6*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val11*val18)))+(f32((val3*val16)))+(f32((val7*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val11*val22)))+(f32((val3*val20)))+(f32((val7*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val11*val26)))+(f32((val3*val24)))+(f32((val7*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val11*val30)))+(f32((val3*val28)))+(f32((val7*val29))));
  }
  var alu23 = ((gidx1*655360)+alu0+(alu1*2560)+alu2+(lidx0*163840)+(lidx1*160)+alu3);
  data0[alu23] = (f16(acc0));
  data0[(alu23+1)] = (f16(acc4));
  data0[(alu23+2)] = (f16(acc8));
  data0[(alu23+3)] = (f16(acc12));
  data0[(alu23+40)] = (f16(acc1));
  data0[(alu23+41)] = (f16(acc5));
  data0[(alu23+42)] = (f16(acc9));
  data0[(alu23+43)] = (f16(acc13));
  data0[(alu23+80)] = (f16(acc2));
  data0[(alu23+81)] = (f16(acc6));
  data0[(alu23+82)] = (f16(acc10));
  data0[(alu23+83)] = (f16(acc14));
  data0[(alu23+120)] = (f16(acc3));
  data0[(alu23+121)] = (f16(acc7));
  data0[(alu23+122)] = (f16(acc11));
  data0[(alu23+123)] = (f16(acc15));
}`;

const r_2_128_5_8_16_80_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 128 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu1 = ((gidx1*1280)+(lidx0*160)+alu0+((ridx0/10)*163840)+((ridx0%10)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+40)];
    var val5 = data2[(alu1+41)];
    var val6 = data2[(alu1+42)];
    var val7 = data2[(alu1+43)];
    var val8 = data2[(alu1+80)];
    var val9 = data2[(alu1+81)];
    var val10 = data2[(alu1+82)];
    var val11 = data2[(alu1+83)];
    var val12 = data2[(alu1+120)];
    var val13 = data2[(alu1+121)];
    var val14 = data2[(alu1+122)];
    var val15 = data2[(alu1+123)];
    var alu2 = ((gidx0*20480)+(lidx1*1280)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+320)];
    var val21 = data3[(alu2+321)];
    var val22 = data3[(alu2+322)];
    var val23 = data3[(alu2+323)];
    var val24 = data3[(alu2+640)];
    var val25 = data3[(alu2+641)];
    var val26 = data3[(alu2+642)];
    var val27 = data3[(alu2+643)];
    var val28 = data3[(alu2+960)];
    var val29 = data3[(alu2+961)];
    var val30 = data3[(alu2+962)];
    var val31 = data3[(alu2+963)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data4[alu22];
  var val33 = data4[(alu22+1)];
  var val34 = data4[(alu22+2)];
  var val35 = data4[(alu22+3)];
  var alu23 = ((gidx1<<5)+alu0+(gidx0<<18)+(lidx0<<2)+(lidx1<<14));
  var val36 = data1[alu23];
  var val37 = data1[(alu23+1)];
  var val38 = data1[(alu23+2)];
  var val39 = data1[(alu23+3)];
  var val40 = data1[(alu23+4096)];
  var val41 = data1[(alu23+4097)];
  var val42 = data1[(alu23+4098)];
  var val43 = data1[(alu23+4099)];
  var val44 = data1[(alu23+8192)];
  var val45 = data1[(alu23+8193)];
  var val46 = data1[(alu23+8194)];
  var val47 = data1[(alu23+8195)];
  var val48 = data1[(alu23+12288)];
  var val49 = data1[(alu23+12289)];
  var val50 = data1[(alu23+12290)];
  var val51 = data1[(alu23+12291)];
  var alu24 = ((gidx1*10240)+alu0+alu20+(lidx0*1280)+alu21);
  data0[alu24] = (val36+(f16(acc0))+val32);
  data0[(alu24+1)] = (val40+(f16(acc1))+val33);
  data0[(alu24+2)] = (val44+(f16(acc2))+val34);
  data0[(alu24+3)] = (val48+(f16(acc3))+val35);
  data0[(alu24+320)] = (val37+(f16(acc4))+val32);
  data0[(alu24+321)] = (val41+(f16(acc5))+val33);
  data0[(alu24+322)] = (val45+(f16(acc6))+val34);
  data0[(alu24+323)] = (val49+(f16(acc7))+val35);
  data0[(alu24+640)] = (val38+(f16(acc8))+val32);
  data0[(alu24+641)] = (val42+(f16(acc9))+val33);
  data0[(alu24+642)] = (val46+(f16(acc10))+val34);
  data0[(alu24+643)] = (val50+(f16(acc11))+val35);
  data0[(alu24+960)] = (val39+(f16(acc12))+val32);
  data0[(alu24+961)] = (val43+(f16(acc13))+val33);
  data0[(alu24+962)] = (val47+(f16(acc14))+val34);
  data0[(alu24+963)] = (val51+(f16(acc15))+val35);
}`;

const r_256_32_80_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu0 = ((gidx0*10240)+(lidx0*320)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+(f32(val3))+(f32(val2))+(f32(val1))+(f32(val0)));
  }
  data0[(lidx0+(gidx0<<5))] = (f16((acc0*0.0031250000465661287f)));
}`;

const r_64_32_80_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu4 = ((gidx0*40960)+(lidx0*1280)+(ridx0<<2));
    var val4 = data1[alu4];
    var val5 = data1[(alu4+1)];
    var val6 = data1[(alu4+2)];
    var val7 = data1[(alu4+3)];
    var val8 = data1[(alu4+320)];
    var val9 = data1[(alu4+321)];
    var val10 = data1[(alu4+322)];
    var val11 = data1[(alu4+323)];
    var val12 = data1[(alu4+640)];
    var val13 = data1[(alu4+641)];
    var val14 = data1[(alu4+642)];
    var val15 = data1[(alu4+643)];
    var val16 = data1[(alu4+960)];
    var val17 = data1[(alu4+961)];
    var val18 = data1[(alu4+962)];
    var val19 = data1[(alu4+963)];
    var alu5 = (val5-val3);
    var alu6 = (val6-val3);
    var alu7 = (val7-val3);
    var alu8 = (val8-val0);
    var alu9 = (val9-val0);
    var alu10 = (val10-val0);
    var alu11 = (val11-val0);
    var alu12 = (val12-val1);
    var alu13 = (val13-val1);
    var alu14 = (val14-val1);
    var alu15 = (val15-val1);
    var alu16 = (val16-val2);
    var alu17 = (val17-val2);
    var alu18 = (val18-val2);
    var alu19 = (val19-val2);
    var alu20 = (val4-val3);
    acc0 = (acc0+(f32((alu7*alu7)))+(f32((alu6*alu6)))+(f32((alu5*alu5)))+(f32((alu20*alu20))));
    acc1 = (acc1+(f32((alu11*alu11)))+(f32((alu10*alu10)))+(f32((alu8*alu8)))+(f32((alu9*alu9))));
    acc2 = (acc2+(f32((alu15*alu15)))+(f32((alu14*alu14)))+(f32((alu12*alu12)))+(f32((alu13*alu13))));
    acc3 = (acc3+(f32((alu19*alu19)))+(f32((alu18*alu18)))+(f32((alu16*alu16)))+(f32((alu17*alu17))));
  }
  data0[alu0] = sqrt((1/((f16((acc0*0.0031250000465661287f)))+(f16(1e-05f)))));
  data0[alu1] = sqrt((1/((f16((acc1*0.0031250000465661287f)))+(f16(1e-05f)))));
  data0[alu2] = sqrt((1/((f16((acc2*0.0031250000465661287f)))+(f16(1e-05f)))));
  data0[alu3] = sqrt((1/((f16((acc3*0.0031250000465661287f)))+(f16(1e-05f)))));
}`;

const E_256_5_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 256 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data2[alu1];
  var val1 = data3[alu1];
  var alu2 = (alu1+1);
  var val2 = data2[alu2];
  var val3 = data3[alu2];
  var alu3 = (alu1+2);
  var val4 = data2[alu3];
  var val5 = data3[alu3];
  var alu4 = (alu1+3);
  var val6 = data2[alu4];
  var val7 = data3[alu4];
  var alu5 = (lidx1<<2);
  var alu6 = (alu0+(gidx1*10240)+(lidx0*1280)+alu5);
  var val8 = data1[alu6];
  var alu7 = (alu6+1);
  var val9 = data1[alu7];
  var alu8 = (alu6+2);
  var val10 = data1[alu8];
  var alu9 = (alu6+3);
  var val11 = data1[alu9];
  var alu10 = (alu6+320);
  var val12 = data1[alu10];
  var alu11 = (alu6+321);
  var val13 = data1[alu11];
  var alu12 = (alu6+322);
  var val14 = data1[alu12];
  var alu13 = (alu6+323);
  var val15 = data1[alu13];
  var alu14 = (alu6+640);
  var val16 = data1[alu14];
  var alu15 = (alu6+641);
  var val17 = data1[alu15];
  var alu16 = (alu6+642);
  var val18 = data1[alu16];
  var alu17 = (alu6+643);
  var val19 = data1[alu17];
  var alu18 = (alu6+960);
  var val20 = data1[alu18];
  var alu19 = (alu6+961);
  var val21 = data1[alu19];
  var alu20 = (alu6+962);
  var val22 = data1[alu20];
  var alu21 = (alu6+963);
  var val23 = data1[alu21];
  var alu22 = (alu0+alu5);
  var val24 = data4[alu22];
  var val25 = data5[alu22];
  var alu23 = (alu22+1);
  var val26 = data4[alu23];
  var val27 = data5[alu23];
  var alu24 = (alu22+2);
  var val28 = data4[alu24];
  var val29 = data5[alu24];
  var alu25 = (alu22+3);
  var val30 = data4[alu25];
  var val31 = data5[alu25];
  data0[alu7] = (val27+(val26*val1*(val9-val0)));
  data0[alu8] = (val29+(val28*val1*(val10-val0)));
  data0[alu9] = (val31+(val30*val1*(val11-val0)));
  data0[alu10] = (val25+(val24*val3*(val12-val2)));
  data0[alu11] = (val27+(val26*val3*(val13-val2)));
  data0[alu12] = (val29+(val28*val3*(val14-val2)));
  data0[alu13] = (val31+(val30*val3*(val15-val2)));
  data0[alu14] = (val25+(val24*val5*(val16-val4)));
  data0[alu15] = (val27+(val26*val5*(val17-val4)));
  data0[alu16] = (val29+(val28*val5*(val18-val4)));
  data0[alu17] = (val31+(val30*val5*(val19-val4)));
  data0[alu18] = (val25+(val24*val7*(val20-val6)));
  data0[alu19] = (val27+(val26*val7*(val21-val6)));
  data0[alu20] = (val29+(val28*val7*(val22-val6)));
  data0[alu21] = (val31+(val30*val7*(val23-val6)));
  data0[alu6] = (val25+(val24*val1*(val8-val0)));
}`;

const r_2_64_77_8_16_10_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 64 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0*40);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 10; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx1*20480)+(gidx2*1310720)+alu0+(lidx1*1280)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+320)];
    var val5 = data1[(alu2+321)];
    var val6 = data1[(alu2+322)];
    var val7 = data1[(alu2+323)];
    var val8 = data1[(alu2+640)];
    var val9 = data1[(alu2+641)];
    var val10 = data1[(alu2+642)];
    var val11 = data1[(alu2+643)];
    var val12 = data1[(alu2+960)];
    var val13 = data1[(alu2+961)];
    var val14 = data1[(alu2+962)];
    var val15 = data1[(alu2+963)];
    var alu3 = ((gidx0*320)+(gidx2*24640)+alu0+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
  }
  var alu9 = (gidx0+(gidx1*4928)+(gidx2*2523136)+(lidx0*315392)+(lidx1*308));
  data0[alu9] = (acc0*0.15811388194561005f);
  data0[(alu9+77)] = (acc1*0.15811388194561005f);
  data0[(alu9+154)] = (acc2*0.15811388194561005f);
  data0[(alu9+231)] = (acc3*0.15811388194561005f);
}`;

const r_2048_32_77 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val0 = data1[((gidx0*2464)+(lidx0*77)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_512_32_77_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu4 = ((gidx0*9856)+(lidx0*308)+ridx0);
    var val4 = data1[alu4];
    var val5 = data1[(alu4+77)];
    var val6 = data1[(alu4+154)];
    var val7 = data1[(alu4+231)];
    acc0 = (acc0+exp2(((val4-val3)*1.4426950408889634f)));
    acc1 = (acc1+exp2(((val5-val0)*1.4426950408889634f)));
    acc2 = (acc2+exp2(((val6-val1)*1.4426950408889634f)));
    acc3 = (acc3+exp2(((val7-val2)*1.4426950408889634f)));
  }
  data0[alu1] = acc1;
  data0[alu2] = acc2;
  data0[alu3] = acc3;
  data0[alu0] = acc0;
}`;

const E_512_77_32_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (gidx0+(gidx1*9856)+(lidx0*308));
  var val0 = data1[alu0];
  var alu1 = (alu0+77);
  var val1 = data1[alu1];
  var alu2 = (alu0+154);
  var val2 = data1[alu2];
  var alu3 = (alu0+231);
  var val3 = data1[alu3];
  var alu4 = ((gidx1<<7)+(lidx0<<2));
  var val4 = data2[alu4];
  var val5 = data3[alu4];
  var alu5 = (alu4+1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = (alu4+2);
  var val8 = data2[alu6];
  var val9 = data3[alu6];
  var alu7 = (alu4+3);
  var val10 = data2[alu7];
  var val11 = data3[alu7];
  data0[alu0] = (f16((exp2(((val0-val4)*1.4426950408889634f))*(1/val5))));
  data0[alu1] = (f16((exp2(((val1-val6)*1.4426950408889634f))*(1/val7))));
  data0[alu2] = (f16((exp2(((val2-val8)*1.4426950408889634f))*(1/val9))));
  data0[alu3] = (f16((exp2(((val3-val10)*1.4426950408889634f))*(1/val11))));
}`;

const r_2_2_64_5_4_16_2_77_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(4,16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 320 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (gidx0/5);
  var alu1 = ((gidx0%5)<<3);
  var alu2 = (lidx2<<2);
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu3 = ((gidx1*1261568)+(gidx2*2523136)+(alu0*4928)+(lidx0*315392)+(lidx1*308)+ridx0);
    var val0 = data1[alu3];
    var val1 = data1[(alu3+77)];
    var val2 = data1[(alu3+154)];
    var val3 = data1[(alu3+231)];
    var alu4 = ((gidx1*160)+(gidx2*24640)+alu1+(lidx0*40)+alu2+(ridx0*320));
    var val4 = data2[alu4];
    var val5 = data2[(alu4+1)];
    var val6 = data2[(alu4+2)];
    var val7 = data2[(alu4+3)];
    acc0 = (acc0+(f32((val4*val0))));
    acc1 = (acc1+(f32((val4*val1))));
    acc2 = (acc2+(f32((val4*val2))));
    acc3 = (acc3+(f32((val4*val3))));
    acc4 = (acc4+(f32((val5*val0))));
    acc5 = (acc5+(f32((val5*val1))));
    acc6 = (acc6+(f32((val5*val2))));
    acc7 = (acc7+(f32((val5*val3))));
    acc8 = (acc8+(f32((val6*val0))));
    acc9 = (acc9+(f32((val6*val1))));
    acc10 = (acc10+(f32((val6*val2))));
    acc11 = (acc11+(f32((val6*val3))));
    acc12 = (acc12+(f32((val7*val0))));
    acc13 = (acc13+(f32((val7*val1))));
    acc14 = (acc14+(f32((val7*val2))));
    acc15 = (acc15+(f32((val7*val3))));
  }
  var alu22 = ((gidx1*655360)+(gidx2*1310720)+(alu0*2560)+alu1+(lidx0*163840)+(lidx1*160)+alu2);
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+40)] = (f16(acc1));
  data0[(alu22+41)] = (f16(acc5));
  data0[(alu22+42)] = (f16(acc9));
  data0[(alu22+43)] = (f16(acc13));
  data0[(alu22+80)] = (f16(acc2));
  data0[(alu22+81)] = (f16(acc6));
  data0[(alu22+82)] = (f16(acc10));
  data0[(alu22+83)] = (f16(acc14));
  data0[(alu22+120)] = (f16(acc3));
  data0[(alu22+121)] = (f16(acc7));
  data0[(alu22+122)] = (f16(acc11));
  data0[(alu22+123)] = (f16(acc15));
}`;

const r_2_128_5_8_16_80_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 128 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu1 = ((gidx1*1280)+(lidx0*160)+alu0+((ridx0/10)*163840)+((ridx0%10)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+40)];
    var val5 = data2[(alu1+41)];
    var val6 = data2[(alu1+42)];
    var val7 = data2[(alu1+43)];
    var val8 = data2[(alu1+80)];
    var val9 = data2[(alu1+81)];
    var val10 = data2[(alu1+82)];
    var val11 = data2[(alu1+83)];
    var val12 = data2[(alu1+120)];
    var val13 = data2[(alu1+121)];
    var val14 = data2[(alu1+122)];
    var val15 = data2[(alu1+123)];
    var alu2 = ((gidx0*20480)+(lidx1*1280)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+320)];
    var val21 = data3[(alu2+321)];
    var val22 = data3[(alu2+322)];
    var val23 = data3[(alu2+323)];
    var val24 = data3[(alu2+640)];
    var val25 = data3[(alu2+641)];
    var val26 = data3[(alu2+642)];
    var val27 = data3[(alu2+643)];
    var val28 = data3[(alu2+960)];
    var val29 = data3[(alu2+961)];
    var val30 = data3[(alu2+962)];
    var val31 = data3[(alu2+963)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = ((gidx1*10240)+alu0+alu20+(lidx0*1280)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+320);
  var val36 = data1[alu26];
  var alu27 = (alu22+321);
  var val37 = data1[alu27];
  var alu28 = (alu22+322);
  var val38 = data1[alu28];
  var alu29 = (alu22+323);
  var val39 = data1[alu29];
  var alu30 = (alu22+640);
  var val40 = data1[alu30];
  var alu31 = (alu22+641);
  var val41 = data1[alu31];
  var alu32 = (alu22+642);
  var val42 = data1[alu32];
  var alu33 = (alu22+643);
  var val43 = data1[alu33];
  var alu34 = (alu22+960);
  var val44 = data1[alu34];
  var alu35 = (alu22+961);
  var val45 = data1[alu35];
  var alu36 = (alu22+962);
  var val46 = data1[alu36];
  var alu37 = (alu22+963);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_256_40_8_16_80_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 256 */
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*20480)+(lidx1*1280)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+320)];
    var val5 = data2[(alu1+321)];
    var val6 = data2[(alu1+322)];
    var val7 = data2[(alu1+323)];
    var val8 = data2[(alu1+640)];
    var val9 = data2[(alu1+641)];
    var val10 = data2[(alu1+642)];
    var val11 = data2[(alu1+643)];
    var val12 = data2[(alu1+960)];
    var val13 = data2[(alu1+961)];
    var val14 = data2[(alu1+962)];
    var val15 = data2[(alu1+963)];
    var alu2 = ((gidx1*10240)+(lidx0*1280)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+320)];
    var val21 = data1[(alu2+321)];
    var val22 = data1[(alu2+322)];
    var val23 = data1[(alu2+323)];
    var val24 = data1[(alu2+640)];
    var val25 = data1[(alu2+641)];
    var val26 = data1[(alu2+642)];
    var val27 = data1[(alu2+643)];
    var val28 = data1[(alu2+960)];
    var val29 = data1[(alu2+961)];
    var val30 = data1[(alu2+962)];
    var val31 = data1[(alu2+963)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = (alu20+(gidx1*81920)+(lidx0*10240)+alu21);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val33);
  data0[(alu23+2)] = ((f16(acc8))+val34);
  data0[(alu23+3)] = ((f16(acc12))+val35);
  data0[(alu23+2560)] = ((f16(acc1))+val32);
  data0[(alu23+2561)] = ((f16(acc5))+val33);
  data0[(alu23+2562)] = ((f16(acc9))+val34);
  data0[(alu23+2563)] = ((f16(acc13))+val35);
  data0[(alu23+5120)] = ((f16(acc2))+val32);
  data0[(alu23+5121)] = ((f16(acc6))+val33);
  data0[(alu23+5122)] = ((f16(acc10))+val34);
  data0[(alu23+5123)] = ((f16(acc14))+val35);
  data0[(alu23+7680)] = ((f16(acc3))+val32);
  data0[(alu23+7681)] = ((f16(acc7))+val33);
  data0[(alu23+7682)] = ((f16(acc11))+val34);
  data0[(alu23+7683)] = ((f16(acc15))+val35);
}`;

const E_1024_20_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 1024 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = (alu0+(gidx1*20480)+(lidx0*2560)+alu1);
  var val0 = data1[alu2];
  var val1 = data1[(alu2+1)];
  var val2 = data1[(alu2+2)];
  var val3 = data1[(alu2+3)];
  var val4 = data1[(alu2+1280)];
  var val5 = data1[(alu2+1281)];
  var val6 = data1[(alu2+1282)];
  var val7 = data1[(alu2+1283)];
  var alu3 = (alu0+(gidx1*10240)+(lidx0*1280)+alu1);
  data0[alu3] = (val0*(1/(exp2(((val4+(val4*val4*val4*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val4);
  data0[(alu3+1)] = (val1*(1/(exp2(((val5+(val5*val5*val5*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val5);
  data0[(alu3+2)] = (val2*(1/(exp2(((val6+(val6*val6*val6*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val6);
  data0[(alu3+3)] = (val3*(1/(exp2(((val7+(val7*val7*val7*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val7);
}`;

const r_256_5_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 256 */
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*81920)+(lidx1*5120)+alu0);
    var val0 = data3[alu1];
    var val1 = data3[(alu1+1)];
    var val2 = data3[(alu1+2)];
    var val3 = data3[(alu1+3)];
    var val4 = data3[(alu1+1280)];
    var val5 = data3[(alu1+1281)];
    var val6 = data3[(alu1+1282)];
    var val7 = data3[(alu1+1283)];
    var val8 = data3[(alu1+2560)];
    var val9 = data3[(alu1+2561)];
    var val10 = data3[(alu1+2562)];
    var val11 = data3[(alu1+2563)];
    var val12 = data3[(alu1+3840)];
    var val13 = data3[(alu1+3841)];
    var val14 = data3[(alu1+3842)];
    var val15 = data3[(alu1+3843)];
    var alu2 = ((gidx1*40960)+(lidx0*5120)+alu0);
    var val16 = data2[alu2];
    var val17 = data2[(alu2+1)];
    var val18 = data2[(alu2+2)];
    var val19 = data2[(alu2+3)];
    var val20 = data2[(alu2+1280)];
    var val21 = data2[(alu2+1281)];
    var val22 = data2[(alu2+1282)];
    var val23 = data2[(alu2+1283)];
    var val24 = data2[(alu2+2560)];
    var val25 = data2[(alu2+2561)];
    var val26 = data2[(alu2+2562)];
    var val27 = data2[(alu2+2563)];
    var val28 = data2[(alu2+3840)];
    var val29 = data2[(alu2+3841)];
    var val30 = data2[(alu2+3842)];
    var val31 = data2[(alu2+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+(gidx1*10240)+(lidx0*1280)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+320);
  var val36 = data1[alu26];
  var alu27 = (alu22+321);
  var val37 = data1[alu27];
  var alu28 = (alu22+322);
  var val38 = data1[alu28];
  var alu29 = (alu22+323);
  var val39 = data1[alu29];
  var alu30 = (alu22+640);
  var val40 = data1[alu30];
  var alu31 = (alu22+641);
  var val41 = data1[alu31];
  var alu32 = (alu22+642);
  var val42 = data1[alu32];
  var alu33 = (alu22+643);
  var val43 = data1[alu33];
  var alu34 = (alu22+960);
  var val44 = data1[alu34];
  var alu35 = (alu22+961);
  var val45 = data1[alu35];
  var alu36 = (alu22+962);
  var val46 = data1[alu36];
  var alu37 = (alu22+963);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_2_10_64_8_16_80_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 10 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*20480)+alu0+(lidx1*1280)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+320)];
    var val5 = data1[(alu2+321)];
    var val6 = data1[(alu2+322)];
    var val7 = data1[(alu2+323)];
    var val8 = data1[(alu2+640)];
    var val9 = data1[(alu2+641)];
    var val10 = data1[(alu2+642)];
    var val11 = data1[(alu2+643)];
    var val12 = data1[(alu2+960)];
    var val13 = data1[(alu2+961)];
    var val14 = data1[(alu2+962)];
    var val15 = data1[(alu2+963)];
    var alu3 = ((gidx1*10240)+(lidx0*1280)+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    var val20 = data2[(alu3+320)];
    var val21 = data2[(alu3+321)];
    var val22 = data2[(alu3+322)];
    var val23 = data2[(alu3+323)];
    var val24 = data2[(alu3+640)];
    var val25 = data2[(alu3+641)];
    var val26 = data2[(alu3+642)];
    var val27 = data2[(alu3+643)];
    var val28 = data2[(alu3+960)];
    var val29 = data2[(alu3+961)];
    var val30 = data2[(alu3+962)];
    var val31 = data2[(alu3+963)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu21 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data3[(alu21+1)];
  var val34 = data3[(alu21+2)];
  var val35 = data3[(alu21+3)];
  var alu22 = ((gidx1<<17)+alu0+(gidx0<<6)+(lidx0<<14)+(lidx1<<2));
  var val36 = data4[alu22];
  var alu23 = (alu22+1);
  var val37 = data4[alu23];
  var alu24 = (alu22+2);
  var val38 = data4[alu24];
  var alu25 = (alu22+3);
  var val39 = data4[alu25];
  var alu26 = (alu22+4096);
  var val40 = data4[alu26];
  var alu27 = (alu22+4097);
  var val41 = data4[alu27];
  var alu28 = (alu22+4098);
  var val42 = data4[alu28];
  var alu29 = (alu22+4099);
  var val43 = data4[alu29];
  var alu30 = (alu22+8192);
  var val44 = data4[alu30];
  var alu31 = (alu22+8193);
  var val45 = data4[alu31];
  var alu32 = (alu22+8194);
  var val46 = data4[alu32];
  var alu33 = (alu22+8195);
  var val47 = data4[alu33];
  var alu34 = (alu22+12288);
  var val48 = data4[alu34];
  var alu35 = (alu22+12289);
  var val49 = data4[alu35];
  var alu36 = (alu22+12290);
  var val50 = data4[alu36];
  var alu37 = (alu22+12291);
  var val51 = data4[alu37];
  data0[alu23] = (val37+(f16(acc4))+val32);
  data0[alu24] = (val38+(f16(acc8))+val32);
  data0[alu25] = (val39+(f16(acc12))+val32);
  data0[alu26] = (val40+(f16(acc1))+val33);
  data0[alu27] = (val41+(f16(acc5))+val33);
  data0[alu28] = (val42+(f16(acc9))+val33);
  data0[alu29] = (val43+(f16(acc13))+val33);
  data0[alu30] = (val44+(f16(acc2))+val34);
  data0[alu31] = (val45+(f16(acc6))+val34);
  data0[alu32] = (val46+(f16(acc10))+val34);
  data0[alu33] = (val47+(f16(acc14))+val34);
  data0[alu34] = (val48+(f16(acc3))+val35);
  data0[alu35] = (val49+(f16(acc7))+val35);
  data0[alu36] = (val50+(f16(acc11))+val35);
  data0[alu37] = (val51+(f16(acc15))+val35);
  data0[alu22] = (val36+(f16(acc0))+val32);
}`;

const r_2_80_2_16_8_320_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = ((lidx1<1)!=true);
  var alu1 = (((gidx0+lidx0)<1)!=true);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu2 = ((gidx1*11520)+(ridx0*9));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+4)];
    var val5 = data2[(alu2+5)];
    var val6 = data2[(alu2+6)];
    var val7 = data2[(alu2+7)];
    var val8 = data2[(alu2+8)];
    var val9 = data2[(alu2+2880)];
    var val10 = data2[(alu2+2881)];
    var val11 = data2[(alu2+2882)];
    var val12 = data2[(alu2+2883)];
    var val13 = data2[(alu2+2884)];
    var val14 = data2[(alu2+2885)];
    var val15 = data2[(alu2+2886)];
    var val16 = data2[(alu2+2887)];
    var val17 = data2[(alu2+2888)];
    var val18 = data2[(alu2+5760)];
    var val19 = data2[(alu2+5761)];
    var val20 = data2[(alu2+5762)];
    var val21 = data2[(alu2+5763)];
    var val22 = data2[(alu2+5764)];
    var val23 = data2[(alu2+5765)];
    var val24 = data2[(alu2+5766)];
    var val25 = data2[(alu2+5767)];
    var val26 = data2[(alu2+5768)];
    var val27 = data2[(alu2+8640)];
    var val28 = data2[(alu2+8641)];
    var val29 = data2[(alu2+8642)];
    var val30 = data2[(alu2+8643)];
    var val31 = data2[(alu2+8644)];
    var val32 = data2[(alu2+8645)];
    var val33 = data2[(alu2+8646)];
    var val34 = data2[(alu2+8647)];
    var val35 = data2[(alu2+8648)];
    var alu3 = ((gidx0<<11)+(lidx0<<7)+(gidx2*1310720)+(ridx0<<12)+(lidx1<<3));
    var val36 = data1[alu3];
    var val37 = select((f16(0.0f)), data1[(alu3+-65)], (alu0&alu1));
    var val38 = select((f16(0.0f)), data1[(alu3+-64)], alu1);
    var val39 = select((f16(0.0f)), data1[(alu3+-63)], alu1);
    var val40 = select((f16(0.0f)), data1[(alu3+-62)], alu1);
    var val41 = select((f16(0.0f)), data1[(alu3+-61)], alu1);
    var val42 = select((f16(0.0f)), data1[(alu3+-60)], alu1);
    var val43 = select((f16(0.0f)), data1[(alu3+-59)], alu1);
    var val44 = select((f16(0.0f)), data1[(alu3+-58)], alu1);
    var val45 = select((f16(0.0f)), data1[(alu3+-57)], alu1);
    var val46 = select((f16(0.0f)), data1[(alu3+-1)], alu0);
    var val47 = data1[(alu3+1)];
    var val48 = data1[(alu3+2)];
    var val49 = data1[(alu3+3)];
    var val50 = data1[(alu3+4)];
    var val51 = data1[(alu3+5)];
    var val52 = data1[(alu3+6)];
    var val53 = data1[(alu3+7)];
    var val54 = select((f16(0.0f)), data1[(alu3+63)], alu0);
    var val55 = data1[(alu3+64)];
    var val56 = data1[(alu3+65)];
    var val57 = data1[(alu3+66)];
    var val58 = data1[(alu3+67)];
    var val59 = data1[(alu3+68)];
    var val60 = data1[(alu3+69)];
    var val61 = data1[(alu3+70)];
    var val62 = data1[(alu3+71)];
    acc0 = (acc0+(f32((val56*val8)))+(f32((val47*val5)))+(f32((val39*val2)))+(f32((val55*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val54*val6)))+(f32((val37*val0)))+(f32((val46*val3))));
    acc1 = (acc1+(f32((val56*val17)))+(f32((val47*val14)))+(f32((val39*val11)))+(f32((val55*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val54*val15)))+(f32((val37*val9)))+(f32((val46*val12))));
    acc2 = (acc2+(f32((val56*val26)))+(f32((val47*val23)))+(f32((val39*val20)))+(f32((val55*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val54*val24)))+(f32((val37*val18)))+(f32((val46*val21))));
    acc3 = (acc3+(f32((val56*val35)))+(f32((val47*val32)))+(f32((val39*val29)))+(f32((val55*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val54*val33)))+(f32((val37*val27)))+(f32((val46*val30))));
    acc4 = (acc4+(f32((val58*val8)))+(f32((val49*val5)))+(f32((val41*val2)))+(f32((val57*val7)))+(f32((val48*val4)))+(f32((val40*val1)))+(f32((val56*val6)))+(f32((val39*val0)))+(f32((val47*val3))));
    acc5 = (acc5+(f32((val58*val17)))+(f32((val49*val14)))+(f32((val41*val11)))+(f32((val57*val16)))+(f32((val48*val13)))+(f32((val40*val10)))+(f32((val56*val15)))+(f32((val39*val9)))+(f32((val47*val12))));
    acc6 = (acc6+(f32((val58*val26)))+(f32((val49*val23)))+(f32((val41*val20)))+(f32((val57*val25)))+(f32((val48*val22)))+(f32((val40*val19)))+(f32((val56*val24)))+(f32((val39*val18)))+(f32((val47*val21))));
    acc7 = (acc7+(f32((val58*val35)))+(f32((val49*val32)))+(f32((val41*val29)))+(f32((val57*val34)))+(f32((val48*val31)))+(f32((val40*val28)))+(f32((val56*val33)))+(f32((val39*val27)))+(f32((val47*val30))));
    acc8 = (acc8+(f32((val60*val8)))+(f32((val51*val5)))+(f32((val43*val2)))+(f32((val59*val7)))+(f32((val50*val4)))+(f32((val42*val1)))+(f32((val58*val6)))+(f32((val41*val0)))+(f32((val49*val3))));
    acc9 = (acc9+(f32((val60*val17)))+(f32((val51*val14)))+(f32((val43*val11)))+(f32((val59*val16)))+(f32((val50*val13)))+(f32((val42*val10)))+(f32((val58*val15)))+(f32((val41*val9)))+(f32((val49*val12))));
    acc10 = (acc10+(f32((val60*val26)))+(f32((val51*val23)))+(f32((val43*val20)))+(f32((val59*val25)))+(f32((val50*val22)))+(f32((val42*val19)))+(f32((val58*val24)))+(f32((val41*val18)))+(f32((val49*val21))));
    acc11 = (acc11+(f32((val60*val35)))+(f32((val51*val32)))+(f32((val43*val29)))+(f32((val59*val34)))+(f32((val50*val31)))+(f32((val42*val28)))+(f32((val58*val33)))+(f32((val41*val27)))+(f32((val49*val30))));
    acc12 = (acc12+(f32((val62*val8)))+(f32((val53*val5)))+(f32((val45*val2)))+(f32((val61*val7)))+(f32((val52*val4)))+(f32((val44*val1)))+(f32((val60*val6)))+(f32((val43*val0)))+(f32((val51*val3))));
    acc13 = (acc13+(f32((val62*val17)))+(f32((val53*val14)))+(f32((val45*val11)))+(f32((val61*val16)))+(f32((val52*val13)))+(f32((val44*val10)))+(f32((val60*val15)))+(f32((val43*val9)))+(f32((val51*val12))));
    acc14 = (acc14+(f32((val62*val26)))+(f32((val53*val23)))+(f32((val45*val20)))+(f32((val61*val25)))+(f32((val52*val22)))+(f32((val44*val19)))+(f32((val60*val24)))+(f32((val43*val18)))+(f32((val51*val21))));
    acc15 = (acc15+(f32((val62*val35)))+(f32((val53*val32)))+(f32((val45*val29)))+(f32((val61*val34)))+(f32((val52*val31)))+(f32((val44*val28)))+(f32((val60*val33)))+(f32((val43*val27)))+(f32((val51*val30))));
  }
  var alu21 = (gidx1<<2);
  var val63 = data3[alu21];
  var val64 = data3[(alu21+1)];
  var val65 = data3[(alu21+2)];
  var val66 = data3[(alu21+3)];
  var alu22 = ((gidx1<<12)+(gidx2*327680)+(gidx0<<9)+(lidx0<<5)+(lidx1<<2));
  data0[alu22] = ((f16(acc0))+val63);
  data0[(alu22+1)] = ((f16(acc4))+val63);
  data0[(alu22+2)] = ((f16(acc8))+val63);
  data0[(alu22+3)] = ((f16(acc12))+val63);
  data0[(alu22+1024)] = ((f16(acc1))+val64);
  data0[(alu22+1025)] = ((f16(acc5))+val64);
  data0[(alu22+1026)] = ((f16(acc9))+val64);
  data0[(alu22+1027)] = ((f16(acc13))+val64);
  data0[(alu22+2048)] = ((f16(acc2))+val65);
  data0[(alu22+2049)] = ((f16(acc6))+val65);
  data0[(alu22+2050)] = ((f16(acc10))+val65);
  data0[(alu22+2051)] = ((f16(acc14))+val65);
  data0[(alu22+3072)] = ((f16(acc3))+val66);
  data0[(alu22+3073)] = ((f16(acc7))+val66);
  data0[(alu22+3074)] = ((f16(acc11))+val66);
  data0[(alu22+3075)] = ((f16(acc15))+val66);
}`;

const r_64_16_640 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var val0 = data1[((gidx0*10240)+(lidx0*640)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*9.765625145519152e-05f)));
  }
}`;

const r_64_16_640n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var val1 = data1[((gidx0*10240)+(lidx0*640)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*9.765625145519152e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_40_16_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/10));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<13)+(gidx2*327680)+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_8_320_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<5);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<4))<31);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu7 = ((gidx1*11520)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+2880)];
    var val10 = data2[(alu7+2881)];
    var val11 = data2[(alu7+2882)];
    var val12 = data2[(alu7+2883)];
    var val13 = data2[(alu7+2884)];
    var val14 = data2[(alu7+2885)];
    var val15 = data2[(alu7+2886)];
    var val16 = data2[(alu7+2887)];
    var val17 = data2[(alu7+2888)];
    var val18 = data2[(alu7+5760)];
    var val19 = data2[(alu7+5761)];
    var val20 = data2[(alu7+5762)];
    var val21 = data2[(alu7+5763)];
    var val22 = data2[(alu7+5764)];
    var val23 = data2[(alu7+5765)];
    var val24 = data2[(alu7+5766)];
    var val25 = data2[(alu7+5767)];
    var val26 = data2[(alu7+5768)];
    var val27 = data2[(alu7+8640)];
    var val28 = data2[(alu7+8641)];
    var val29 = data2[(alu7+8642)];
    var val30 = data2[(alu7+8643)];
    var val31 = data2[(alu7+8644)];
    var val32 = data2[(alu7+8645)];
    var val33 = data2[(alu7+8646)];
    var val34 = data2[(alu7+8647)];
    var val35 = data2[(alu7+8648)];
    var alu8 = (alu0+alu1+(gidx2*327680)+(ridx0<<10)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-33)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-32)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-31)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-30)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-29)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-28)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+31)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+32)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+33)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+34)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+35)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+36)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx1<<12)+(gidx2*655360)+alu0+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+1024)] = (val57+(f16(acc1))+val56);
  data0[(alu30+1025)] = (val57+(f16(acc5))+val56);
  data0[(alu30+1026)] = (val57+(f16(acc9))+val56);
  data0[(alu30+1027)] = (val57+(f16(acc13))+val56);
  data0[(alu30+2048)] = (val59+(f16(acc2))+val58);
  data0[(alu30+2049)] = (val59+(f16(acc6))+val58);
  data0[(alu30+2050)] = (val59+(f16(acc10))+val58);
  data0[(alu30+2051)] = (val59+(f16(acc14))+val58);
  data0[(alu30+3072)] = (val61+(f16(acc3))+val60);
  data0[(alu30+3073)] = (val61+(f16(acc7))+val60);
  data0[(alu30+3074)] = (val61+(f16(acc11))+val60);
  data0[(alu30+3075)] = (val61+(f16(acc15))+val60);
}`;

const r_64_16_1280 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var val0 = data1[((gidx0*20480)+(lidx0*1280)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*4.882812572759576e-05f)));
  }
}`;

const r_64_16_1280n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var val1 = data1[((gidx0*20480)+(lidx0*1280)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*4.882812572759576e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_80_16_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/20));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<13)+(gidx2*655360)+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_8_640_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx2*655360);
  var alu1 = (gidx0<<9);
  var alu2 = (lidx0<<5);
  var alu3 = (lidx1<<2);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (((gidx0+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx0<<4))<31);
  var alu7 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu8 = ((gidx1*23040)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+5760)];
    var val10 = data2[(alu8+5761)];
    var val11 = data2[(alu8+5762)];
    var val12 = data2[(alu8+5763)];
    var val13 = data2[(alu8+5764)];
    var val14 = data2[(alu8+5765)];
    var val15 = data2[(alu8+5766)];
    var val16 = data2[(alu8+5767)];
    var val17 = data2[(alu8+5768)];
    var val18 = data2[(alu8+11520)];
    var val19 = data2[(alu8+11521)];
    var val20 = data2[(alu8+11522)];
    var val21 = data2[(alu8+11523)];
    var val22 = data2[(alu8+11524)];
    var val23 = data2[(alu8+11525)];
    var val24 = data2[(alu8+11526)];
    var val25 = data2[(alu8+11527)];
    var val26 = data2[(alu8+11528)];
    var val27 = data2[(alu8+17280)];
    var val28 = data2[(alu8+17281)];
    var val29 = data2[(alu8+17282)];
    var val30 = data2[(alu8+17283)];
    var val31 = data2[(alu8+17284)];
    var val32 = data2[(alu8+17285)];
    var val33 = data2[(alu8+17286)];
    var val34 = data2[(alu8+17287)];
    var val35 = data2[(alu8+17288)];
    var alu9 = (alu1+alu2+alu0+(ridx0<<10)+alu3);
    var val36 = data1[alu9];
    var val37 = select((f16(0.0f)), data1[(alu9+-33)], (alu4&alu5));
    var val38 = select((f16(0.0f)), data1[(alu9+-32)], alu5);
    var val39 = select((f16(0.0f)), data1[(alu9+-31)], alu5);
    var val40 = select((f16(0.0f)), data1[(alu9+-30)], alu5);
    var val41 = select((f16(0.0f)), data1[(alu9+-29)], alu5);
    var val42 = select((f16(0.0f)), data1[(alu9+-28)], (alu7&alu5));
    var val43 = select((f16(0.0f)), data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select((f16(0.0f)), data1[(alu9+4)], alu7);
    var val48 = select((f16(0.0f)), data1[(alu9+31)], (alu6&alu4));
    var val49 = select((f16(0.0f)), data1[(alu9+32)], alu6);
    var val50 = select((f16(0.0f)), data1[(alu9+33)], alu6);
    var val51 = select((f16(0.0f)), data1[(alu9+34)], alu6);
    var val52 = select((f16(0.0f)), data1[(alu9+35)], alu6);
    var val53 = select((f16(0.0f)), data1[(alu9+36)], (alu7&alu6));
    acc0 = (acc0+(f32((val8*val50)))+(f32((val5*val44)))+(f32((val2*val39)))+(f32((val7*val49)))+(f32((val4*val36)))+(f32((val1*val38)))+(f32((val6*val48)))+(f32((val3*val43)))+(f32((val0*val37))));
    acc1 = (acc1+(f32((val8*val51)))+(f32((val5*val45)))+(f32((val2*val40)))+(f32((val7*val50)))+(f32((val4*val44)))+(f32((val1*val39)))+(f32((val6*val49)))+(f32((val3*val36)))+(f32((val0*val38))));
    acc2 = (acc2+(f32((val8*val52)))+(f32((val5*val46)))+(f32((val2*val41)))+(f32((val7*val51)))+(f32((val4*val45)))+(f32((val1*val40)))+(f32((val6*val50)))+(f32((val3*val44)))+(f32((val0*val39))));
    acc3 = (acc3+(f32((val8*val53)))+(f32((val5*val47)))+(f32((val2*val42)))+(f32((val7*val52)))+(f32((val4*val46)))+(f32((val1*val41)))+(f32((val6*val51)))+(f32((val3*val45)))+(f32((val0*val40))));
    acc4 = (acc4+(f32((val17*val50)))+(f32((val14*val44)))+(f32((val11*val39)))+(f32((val16*val49)))+(f32((val13*val36)))+(f32((val10*val38)))+(f32((val15*val48)))+(f32((val9*val37)))+(f32((val12*val43))));
    acc5 = (acc5+(f32((val17*val51)))+(f32((val14*val45)))+(f32((val11*val40)))+(f32((val16*val50)))+(f32((val13*val44)))+(f32((val10*val39)))+(f32((val15*val49)))+(f32((val9*val38)))+(f32((val12*val36))));
    acc6 = (acc6+(f32((val17*val52)))+(f32((val14*val46)))+(f32((val11*val41)))+(f32((val16*val51)))+(f32((val13*val45)))+(f32((val10*val40)))+(f32((val15*val50)))+(f32((val9*val39)))+(f32((val12*val44))));
    acc7 = (acc7+(f32((val17*val53)))+(f32((val14*val47)))+(f32((val11*val42)))+(f32((val16*val52)))+(f32((val13*val46)))+(f32((val10*val41)))+(f32((val15*val51)))+(f32((val9*val40)))+(f32((val12*val45))));
    acc8 = (acc8+(f32((val26*val50)))+(f32((val23*val44)))+(f32((val20*val39)))+(f32((val25*val49)))+(f32((val22*val36)))+(f32((val19*val38)))+(f32((val24*val48)))+(f32((val18*val37)))+(f32((val21*val43))));
    acc9 = (acc9+(f32((val26*val51)))+(f32((val23*val45)))+(f32((val20*val40)))+(f32((val25*val50)))+(f32((val22*val44)))+(f32((val19*val39)))+(f32((val24*val49)))+(f32((val18*val38)))+(f32((val21*val36))));
    acc10 = (acc10+(f32((val26*val52)))+(f32((val23*val46)))+(f32((val20*val41)))+(f32((val25*val51)))+(f32((val22*val45)))+(f32((val19*val40)))+(f32((val24*val50)))+(f32((val18*val39)))+(f32((val21*val44))));
    acc11 = (acc11+(f32((val26*val53)))+(f32((val23*val47)))+(f32((val20*val42)))+(f32((val25*val52)))+(f32((val22*val46)))+(f32((val19*val41)))+(f32((val24*val51)))+(f32((val18*val40)))+(f32((val21*val45))));
    acc12 = (acc12+(f32((val35*val50)))+(f32((val32*val44)))+(f32((val29*val39)))+(f32((val34*val49)))+(f32((val31*val36)))+(f32((val28*val38)))+(f32((val33*val48)))+(f32((val27*val37)))+(f32((val30*val43))));
    acc13 = (acc13+(f32((val35*val51)))+(f32((val32*val45)))+(f32((val29*val40)))+(f32((val34*val50)))+(f32((val31*val44)))+(f32((val28*val39)))+(f32((val33*val49)))+(f32((val27*val38)))+(f32((val30*val36))));
    acc14 = (acc14+(f32((val35*val52)))+(f32((val32*val46)))+(f32((val29*val41)))+(f32((val34*val51)))+(f32((val31*val45)))+(f32((val28*val40)))+(f32((val33*val50)))+(f32((val27*val39)))+(f32((val30*val44))));
    acc15 = (acc15+(f32((val35*val53)))+(f32((val32*val47)))+(f32((val29*val42)))+(f32((val34*val52)))+(f32((val31*val46)))+(f32((val28*val41)))+(f32((val33*val51)))+(f32((val27*val40)))+(f32((val30*val45))));
  }
  var alu27 = ((gidx1<<12)+alu0+alu1+alu2+alu3);
  data0[alu27] = acc0;
  data0[(alu27+1)] = acc1;
  data0[(alu27+2)] = acc2;
  data0[(alu27+3)] = acc3;
  data0[(alu27+1024)] = acc4;
  data0[(alu27+1025)] = acc5;
  data0[(alu27+1026)] = acc6;
  data0[(alu27+1027)] = acc7;
  data0[(alu27+2048)] = acc8;
  data0[(alu27+2049)] = acc9;
  data0[(alu27+2050)] = acc10;
  data0[(alu27+2051)] = acc11;
  data0[(alu27+3072)] = acc12;
  data0[(alu27+3073)] = acc13;
  data0[(alu27+3074)] = acc14;
  data0[(alu27+3075)] = acc15;
}`;

const r_2_20_16_8_16_80_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 20 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu2 = ((gidx1*10240)+(lidx0*1280)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+320)];
    var val5 = data2[(alu2+321)];
    var val6 = data2[(alu2+322)];
    var val7 = data2[(alu2+323)];
    var val8 = data2[(alu2+640)];
    var val9 = data2[(alu2+641)];
    var val10 = data2[(alu2+642)];
    var val11 = data2[(alu2+643)];
    var val12 = data2[(alu2+960)];
    var val13 = data2[(alu2+961)];
    var val14 = data2[(alu2+962)];
    var val15 = data2[(alu2+963)];
    var alu3 = (alu0+(gidx2*327680)+alu1+(ridx0<<12));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+1024)];
    var val21 = data1[(alu3+1025)];
    var val22 = data1[(alu3+1026)];
    var val23 = data1[(alu3+1027)];
    var val24 = data1[(alu3+2048)];
    var val25 = data1[(alu3+2049)];
    var val26 = data1[(alu3+2050)];
    var val27 = data1[(alu3+2051)];
    var val28 = data1[(alu3+3072)];
    var val29 = data1[(alu3+3073)];
    var val30 = data1[(alu3+3074)];
    var val31 = data1[(alu3+3075)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<15)+(gidx2*655360)+alu0+(lidx0<<12)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+1024);
  var val44 = data4[alu29];
  var alu30 = (alu25+1025);
  var val45 = data4[alu30];
  var alu31 = (alu25+1026);
  var val46 = data4[alu31];
  var alu32 = (alu25+1027);
  var val47 = data4[alu32];
  var alu33 = (alu25+2048);
  var val48 = data4[alu33];
  var alu34 = (alu25+2049);
  var val49 = data4[alu34];
  var alu35 = (alu25+2050);
  var val50 = data4[alu35];
  var alu36 = (alu25+2051);
  var val51 = data4[alu36];
  var alu37 = (alu25+3072);
  var val52 = data4[alu37];
  var alu38 = (alu25+3073);
  var val53 = data4[alu38];
  var alu39 = (alu25+3074);
  var val54 = data4[alu39];
  var alu40 = (alu25+3075);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const E_2_80_16_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/20));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<13)+(gidx2*655360)+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  data0[alu3] = (val1+(val0*val3*(val5-val2)));
  data0[alu4] = (val1+(val0*val3*(val6-val2)));
  data0[alu5] = (val1+(val0*val3*(val7-val2)));
  data0[alu2] = (val1+(val0*val3*(val4-val2)));
}`;

const r_2_20_16_8_16_160_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 20 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*655360);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx1<<2);
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu3 = ((gidx1*20480)+(lidx0*2560)+(ridx0<<2));
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+640)];
    var val5 = data2[(alu3+641)];
    var val6 = data2[(alu3+642)];
    var val7 = data2[(alu3+643)];
    var val8 = data2[(alu3+1280)];
    var val9 = data2[(alu3+1281)];
    var val10 = data2[(alu3+1282)];
    var val11 = data2[(alu3+1283)];
    var val12 = data2[(alu3+1920)];
    var val13 = data2[(alu3+1921)];
    var val14 = data2[(alu3+1922)];
    var val15 = data2[(alu3+1923)];
    var alu4 = (alu1+alu0+alu2+(ridx0<<12));
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+1024)];
    var val21 = data1[(alu4+1025)];
    var val22 = data1[(alu4+1026)];
    var val23 = data1[(alu4+1027)];
    var val24 = data1[(alu4+2048)];
    var val25 = data1[(alu4+2049)];
    var val26 = data1[(alu4+2050)];
    var val27 = data1[(alu4+2051)];
    var val28 = data1[(alu4+3072)];
    var val29 = data1[(alu4+3073)];
    var val30 = data1[(alu4+3074)];
    var val31 = data1[(alu4+3075)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
  }
  var alu22 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = ((gidx1<<15)+alu0+alu1+(lidx0<<12)+alu2);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val32);
  data0[(alu23+2)] = ((f16(acc8))+val32);
  data0[(alu23+3)] = ((f16(acc12))+val32);
  data0[(alu23+1024)] = ((f16(acc1))+val33);
  data0[(alu23+1025)] = ((f16(acc5))+val33);
  data0[(alu23+1026)] = ((f16(acc9))+val33);
  data0[(alu23+1027)] = ((f16(acc13))+val33);
  data0[(alu23+2048)] = ((f16(acc2))+val34);
  data0[(alu23+2049)] = ((f16(acc6))+val34);
  data0[(alu23+2050)] = ((f16(acc10))+val34);
  data0[(alu23+2051)] = ((f16(acc14))+val34);
  data0[(alu23+3072)] = ((f16(acc3))+val35);
  data0[(alu23+3073)] = ((f16(acc7))+val35);
  data0[(alu23+3074)] = ((f16(acc11))+val35);
  data0[(alu23+3075)] = ((f16(acc15))+val35);
}`;

const r_2_1024_16_40 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var val0 = data1[(gidx0+(gidx1*655360)+(lidx0*40960)+(ridx0<<10))];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[(gidx0+(gidx1<<10))] = (f16((acc1*0.0015625000232830644f)));
  }
}`;

const r_2_1024_16_40n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0+(gidx1<<10));
  var val0 = data2[alu0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var val1 = data1[(gidx0+(gidx1*655360)+(lidx0*40960)+(ridx0<<10))];
    var alu1 = (val1-val0);
    acc0 = (acc0+(f32((alu1*alu1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[alu0] = sqrt((1/((f16((acc1*0.0015625000232830644f)))+(f16(1e-05f)))));
  }
}`;

const E_2_32_10_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*655360);
  var alu1 = (gidx0<<6);
  var alu2 = (gidx1<<5);
  var alu3 = (lidx0<<2);
  var alu4 = (alu2+(gidx2<<10)+alu3);
  var val0 = data2[alu4];
  var val1 = data3[alu4];
  var alu5 = (alu4+1);
  var val2 = data2[alu5];
  var val3 = data3[alu5];
  var alu6 = (alu4+2);
  var val4 = data2[alu6];
  var val5 = data3[alu6];
  var alu7 = (alu4+3);
  var val6 = data2[alu7];
  var val7 = data3[alu7];
  var alu8 = (lidx1<<2);
  var alu9 = (alu1+alu8);
  var val8 = data4[alu9];
  var val9 = data5[alu9];
  var alu10 = (alu9+1);
  var val10 = data4[alu10];
  var val11 = data5[alu10];
  var alu11 = (alu9+2);
  var val12 = data4[alu11];
  var val13 = data5[alu11];
  var alu12 = (alu9+3);
  var val14 = data4[alu12];
  var val15 = data5[alu12];
  var alu13 = (alu2+alu0+(gidx0<<16)+alu3+(lidx1<<12));
  var val16 = data1[alu13];
  var val17 = data1[(alu13+1)];
  var val18 = data1[(alu13+2)];
  var val19 = data1[(alu13+3)];
  var val20 = data1[(alu13+1024)];
  var val21 = data1[(alu13+1025)];
  var val22 = data1[(alu13+1026)];
  var val23 = data1[(alu13+1027)];
  var val24 = data1[(alu13+2048)];
  var val25 = data1[(alu13+2049)];
  var val26 = data1[(alu13+2050)];
  var val27 = data1[(alu13+2051)];
  var val28 = data1[(alu13+3072)];
  var val29 = data1[(alu13+3073)];
  var val30 = data1[(alu13+3074)];
  var val31 = data1[(alu13+3075)];
  var alu14 = ((gidx1*20480)+alu0+alu1+(lidx0*2560)+alu8);
  data0[(alu14+640)] = (val9+(val8*val3*(val17-val2)));
  data0[(alu14+1280)] = (val9+(val8*val5*(val18-val4)));
  data0[(alu14+1920)] = (val9+(val8*val7*(val19-val6)));
  data0[(alu14+1)] = (val11+(val10*val1*(val20-val0)));
  data0[(alu14+641)] = (val11+(val10*val3*(val21-val2)));
  data0[(alu14+1281)] = (val11+(val10*val5*(val22-val4)));
  data0[(alu14+1921)] = (val11+(val10*val7*(val23-val6)));
  data0[(alu14+2)] = (val13+(val12*val1*(val24-val0)));
  data0[(alu14+642)] = (val13+(val12*val3*(val25-val2)));
  data0[(alu14+1282)] = (val13+(val12*val5*(val26-val4)));
  data0[(alu14+1922)] = (val13+(val12*val7*(val27-val6)));
  data0[(alu14+3)] = (val15+(val14*val1*(val28-val0)));
  data0[(alu14+643)] = (val15+(val14*val3*(val29-val2)));
  data0[(alu14+1283)] = (val15+(val14*val5*(val30-val4)));
  data0[(alu14+1923)] = (val15+(val14*val7*(val31-val6)));
  data0[alu14] = (val9+(val8*val1*(val16-val0)));
}`;

const r_64_10_8_16_160_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1*20480);
  var alu1 = (lidx0*2560);
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu2 = (ridx0<<2);
    var alu3 = ((gidx0*40960)+(lidx1*2560)+alu2);
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+640)];
    var val5 = data2[(alu3+641)];
    var val6 = data2[(alu3+642)];
    var val7 = data2[(alu3+643)];
    var val8 = data2[(alu3+1280)];
    var val9 = data2[(alu3+1281)];
    var val10 = data2[(alu3+1282)];
    var val11 = data2[(alu3+1283)];
    var val12 = data2[(alu3+1920)];
    var val13 = data2[(alu3+1921)];
    var val14 = data2[(alu3+1922)];
    var val15 = data2[(alu3+1923)];
    var alu4 = (alu0+alu1+alu2);
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+640)];
    var val21 = data1[(alu4+641)];
    var val22 = data1[(alu4+642)];
    var val23 = data1[(alu4+643)];
    var val24 = data1[(alu4+1280)];
    var val25 = data1[(alu4+1281)];
    var val26 = data1[(alu4+1282)];
    var val27 = data1[(alu4+1283)];
    var val28 = data1[(alu4+1920)];
    var val29 = data1[(alu4+1921)];
    var val30 = data1[(alu4+1922)];
    var val31 = data1[(alu4+1923)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu22 = ((gidx0<<6)+alu0+alu1+(lidx1<<2));
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+640)] = (f16(acc1));
  data0[(alu22+641)] = (f16(acc5));
  data0[(alu22+642)] = (f16(acc9));
  data0[(alu22+643)] = (f16(acc13));
  data0[(alu22+1280)] = (f16(acc2));
  data0[(alu22+1281)] = (f16(acc6));
  data0[(alu22+1282)] = (f16(acc10));
  data0[(alu22+1283)] = (f16(acc14));
  data0[(alu22+1920)] = (f16(acc3));
  data0[(alu22+1921)] = (f16(acc7));
  data0[(alu22+1922)] = (f16(acc11));
  data0[(alu22+1923)] = (f16(acc15));
}`;

const r_2_8_32_16_8_16_20_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0>>4);
  var alu1 = (gidx0&15);
  var alu2 = ((gidx1*80)+(gidx2*655360));
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
  for (var ridx0 = 0; ridx0 < 20; ridx0++) {
    var alu3 = (ridx0<<2);
    var alu4 = (alu2+(alu0*20480)+(lidx0*2560)+alu3);
    var val0 = data1[alu4];
    var val1 = data1[(alu4+1)];
    var val2 = data1[(alu4+2)];
    var val3 = data1[(alu4+3)];
    var val4 = data1[(alu4+640)];
    var val5 = data1[(alu4+641)];
    var val6 = data1[(alu4+642)];
    var val7 = data1[(alu4+643)];
    var val8 = data1[(alu4+1280)];
    var val9 = data1[(alu4+1281)];
    var val10 = data1[(alu4+1282)];
    var val11 = data1[(alu4+1283)];
    var val12 = data1[(alu4+1920)];
    var val13 = data1[(alu4+1921)];
    var val14 = data1[(alu4+1922)];
    var val15 = data1[(alu4+1923)];
    var alu5 = (alu2+(alu1*40960)+(lidx1*2560)+alu3);
    var val16 = data2[alu5];
    var val17 = data2[(alu5+1)];
    var val18 = data2[(alu5+2)];
    var val19 = data2[(alu5+3)];
    var val20 = data2[(alu5+640)];
    var val21 = data2[(alu5+641)];
    var val22 = data2[(alu5+642)];
    var val23 = data2[(alu5+643)];
    var val24 = data2[(alu5+1280)];
    var val25 = data2[(alu5+1281)];
    var val26 = data2[(alu5+1282)];
    var val27 = data2[(alu5+1283)];
    var val28 = data2[(alu5+1920)];
    var val29 = data2[(alu5+1921)];
    var val30 = data2[(alu5+1922)];
    var val31 = data2[(alu5+1923)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc2 = (acc2+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc3 = (acc3+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc4 = (acc4+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc7 = (acc7+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc8 = (acc8+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc9 = (acc9+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc12 = (acc12+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc13 = (acc13+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc14 = (acc14+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu23 = ((gidx1<<20)+(gidx2<<23)+(alu0<<15)+(alu1<<6)+(lidx0<<12)+(lidx1<<2));
  data0[alu23] = (acc0*0.11180339753627777f);
  data0[(alu23+1)] = (acc4*0.11180339753627777f);
  data0[(alu23+2)] = (acc8*0.11180339753627777f);
  data0[(alu23+3)] = (acc12*0.11180339753627777f);
  data0[(alu23+1024)] = (acc1*0.11180339753627777f);
  data0[(alu23+1025)] = (acc5*0.11180339753627777f);
  data0[(alu23+1026)] = (acc9*0.11180339753627777f);
  data0[(alu23+1027)] = (acc13*0.11180339753627777f);
  data0[(alu23+2048)] = (acc2*0.11180339753627777f);
  data0[(alu23+2049)] = (acc6*0.11180339753627777f);
  data0[(alu23+2050)] = (acc10*0.11180339753627777f);
  data0[(alu23+2051)] = (acc14*0.11180339753627777f);
  data0[(alu23+3072)] = (acc3*0.11180339753627777f);
  data0[(alu23+3073)] = (acc7*0.11180339753627777f);
  data0[(alu23+3074)] = (acc11*0.11180339753627777f);
  data0[(alu23+3075)] = (acc15*0.11180339753627777f);
}`;

const r_512_32_256_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu0 = ((gidx0<<15)+(lidx0<<10)+(ridx0<<2));
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

const r_128_32_256_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
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
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu4 = ((gidx0<<17)+(lidx0<<12)+(ridx0<<2));
    var val4 = data1[alu4];
    var val5 = data1[(alu4+1)];
    var val6 = data1[(alu4+2)];
    var val7 = data1[(alu4+3)];
    var val8 = data1[(alu4+1024)];
    var val9 = data1[(alu4+1025)];
    var val10 = data1[(alu4+1026)];
    var val11 = data1[(alu4+1027)];
    var val12 = data1[(alu4+2048)];
    var val13 = data1[(alu4+2049)];
    var val14 = data1[(alu4+2050)];
    var val15 = data1[(alu4+2051)];
    var val16 = data1[(alu4+3072)];
    var val17 = data1[(alu4+3073)];
    var val18 = data1[(alu4+3074)];
    var val19 = data1[(alu4+3075)];
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

const E_2048_16_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 2048 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<13)+(lidx0<<10)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (1/val1);
  data0[alu2] = (f16((exp2(((val3-val0)*1.4426950408889634f))*alu5)));
  data0[alu3] = (f16((exp2(((val4-val0)*1.4426950408889634f))*alu5)));
  data0[alu4] = (f16((exp2(((val5-val0)*1.4426950408889634f))*alu5)));
  data0[alu1] = (f16((exp2(((val2-val0)*1.4426950408889634f))*alu5)));
}`;

const r_2_4_16_5_2_16_4_256_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 80 */
  var gidx1 = i32(gindex.y); /* 4 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (gidx2*655360);
  var alu1 = (gidx0/5);
  var alu2 = ((gidx0%5)<<4);
  var alu3 = (lidx2<<2);
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
    var alu4 = ((gidx1*160)+alu0+alu2+(lidx0*80)+alu3+(ridx0*2560));
    var val0 = data2[alu4];
    var val1 = data2[(alu4+1)];
    var val2 = data2[(alu4+2)];
    var val3 = data2[(alu4+3)];
    var val4 = data2[(alu4+640)];
    var val5 = data2[(alu4+641)];
    var val6 = data2[(alu4+642)];
    var val7 = data2[(alu4+643)];
    var val8 = data2[(alu4+1280)];
    var val9 = data2[(alu4+1281)];
    var val10 = data2[(alu4+1282)];
    var val11 = data2[(alu4+1283)];
    var val12 = data2[(alu4+1920)];
    var val13 = data2[(alu4+1921)];
    var val14 = data2[(alu4+1922)];
    var val15 = data2[(alu4+1923)];
    var alu5 = ((gidx1<<21)+(gidx2<<23)+(alu1<<16)+(lidx0<<20)+(lidx1<<12)+(ridx0<<2));
    var val16 = data1[alu5];
    var val17 = data1[(alu5+1)];
    var val18 = data1[(alu5+2)];
    var val19 = data1[(alu5+3)];
    var val20 = data1[(alu5+1024)];
    var val21 = data1[(alu5+1025)];
    var val22 = data1[(alu5+1026)];
    var val23 = data1[(alu5+1027)];
    var val24 = data1[(alu5+2048)];
    var val25 = data1[(alu5+2049)];
    var val26 = data1[(alu5+2050)];
    var val27 = data1[(alu5+2051)];
    var val28 = data1[(alu5+3072)];
    var val29 = data1[(alu5+3073)];
    var val30 = data1[(alu5+3074)];
    var val31 = data1[(alu5+3075)];
    acc0 = (acc0+(f32((val12*val19)))+(f32((val8*val18)))+(f32((val4*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val12*val23)))+(f32((val8*val22)))+(f32((val4*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val12*val27)))+(f32((val8*val26)))+(f32((val4*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val12*val31)))+(f32((val8*val30)))+(f32((val4*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val13*val19)))+(f32((val9*val18)))+(f32((val1*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val13*val23)))+(f32((val9*val22)))+(f32((val1*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val13*val27)))+(f32((val9*val26)))+(f32((val1*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val13*val31)))+(f32((val9*val30)))+(f32((val1*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val14*val19)))+(f32((val10*val18)))+(f32((val2*val16)))+(f32((val6*val17))));
    acc9 = (acc9+(f32((val14*val23)))+(f32((val10*val22)))+(f32((val2*val20)))+(f32((val6*val21))));
    acc10 = (acc10+(f32((val14*val27)))+(f32((val10*val26)))+(f32((val2*val24)))+(f32((val6*val25))));
    acc11 = (acc11+(f32((val14*val31)))+(f32((val10*val30)))+(f32((val2*val28)))+(f32((val6*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val11*val18)))+(f32((val3*val16)))+(f32((val7*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val11*val22)))+(f32((val3*val20)))+(f32((val7*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val11*val26)))+(f32((val3*val24)))+(f32((val7*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val11*val30)))+(f32((val3*val28)))+(f32((val7*val29))));
  }
  var alu23 = ((gidx1*163840)+alu0+(alu1*5120)+alu2+(lidx0*81920)+(lidx1*320)+alu3);
  data0[alu23] = (f16(acc0));
  data0[(alu23+1)] = (f16(acc4));
  data0[(alu23+2)] = (f16(acc8));
  data0[(alu23+3)] = (f16(acc12));
  data0[(alu23+80)] = (f16(acc1));
  data0[(alu23+81)] = (f16(acc5));
  data0[(alu23+82)] = (f16(acc9));
  data0[(alu23+83)] = (f16(acc13));
  data0[(alu23+160)] = (f16(acc2));
  data0[(alu23+161)] = (f16(acc6));
  data0[(alu23+162)] = (f16(acc10));
  data0[(alu23+163)] = (f16(acc14));
  data0[(alu23+240)] = (f16(acc3));
  data0[(alu23+241)] = (f16(acc7));
  data0[(alu23+242)] = (f16(acc11));
  data0[(alu23+243)] = (f16(acc15));
}`;

const r_2_32_10_8_16_160_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*655360);
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu1 = ((gidx1*2560)+(lidx0*320)+alu0+((ridx0/20)*81920)+((ridx0%20)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+80)];
    var val5 = data2[(alu1+81)];
    var val6 = data2[(alu1+82)];
    var val7 = data2[(alu1+83)];
    var val8 = data2[(alu1+160)];
    var val9 = data2[(alu1+161)];
    var val10 = data2[(alu1+162)];
    var val11 = data2[(alu1+163)];
    var val12 = data2[(alu1+240)];
    var val13 = data2[(alu1+241)];
    var val14 = data2[(alu1+242)];
    var val15 = data2[(alu1+243)];
    var alu2 = ((gidx0*40960)+(lidx1*2560)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+640)];
    var val21 = data3[(alu2+641)];
    var val22 = data3[(alu2+642)];
    var val23 = data3[(alu2+643)];
    var val24 = data3[(alu2+1280)];
    var val25 = data3[(alu2+1281)];
    var val26 = data3[(alu2+1282)];
    var val27 = data3[(alu2+1283)];
    var val28 = data3[(alu2+1920)];
    var val29 = data3[(alu2+1921)];
    var val30 = data3[(alu2+1922)];
    var val31 = data3[(alu2+1923)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data4[alu22];
  var val33 = data4[(alu22+1)];
  var val34 = data4[(alu22+2)];
  var val35 = data4[(alu22+3)];
  var alu23 = ((gidx1<<5)+alu0+(gidx0<<16)+(lidx0<<2)+(lidx1<<12));
  var val36 = data1[alu23];
  var val37 = data1[(alu23+1)];
  var val38 = data1[(alu23+2)];
  var val39 = data1[(alu23+3)];
  var val40 = data1[(alu23+1024)];
  var val41 = data1[(alu23+1025)];
  var val42 = data1[(alu23+1026)];
  var val43 = data1[(alu23+1027)];
  var val44 = data1[(alu23+2048)];
  var val45 = data1[(alu23+2049)];
  var val46 = data1[(alu23+2050)];
  var val47 = data1[(alu23+2051)];
  var val48 = data1[(alu23+3072)];
  var val49 = data1[(alu23+3073)];
  var val50 = data1[(alu23+3074)];
  var val51 = data1[(alu23+3075)];
  var alu24 = ((gidx1*20480)+alu0+alu20+(lidx0*2560)+alu21);
  data0[alu24] = (val36+(f16(acc0))+val32);
  data0[(alu24+1)] = (val40+(f16(acc4))+val33);
  data0[(alu24+2)] = (val44+(f16(acc8))+val34);
  data0[(alu24+3)] = (val48+(f16(acc12))+val35);
  data0[(alu24+640)] = (val37+(f16(acc1))+val32);
  data0[(alu24+641)] = (val41+(f16(acc5))+val33);
  data0[(alu24+642)] = (val45+(f16(acc9))+val34);
  data0[(alu24+643)] = (val49+(f16(acc13))+val35);
  data0[(alu24+1280)] = (val38+(f16(acc2))+val32);
  data0[(alu24+1281)] = (val42+(f16(acc6))+val33);
  data0[(alu24+1282)] = (val46+(f16(acc10))+val34);
  data0[(alu24+1283)] = (val50+(f16(acc14))+val35);
  data0[(alu24+1920)] = (val39+(f16(acc3))+val32);
  data0[(alu24+1921)] = (val43+(f16(acc7))+val33);
  data0[(alu24+1922)] = (val47+(f16(acc11))+val34);
  data0[(alu24+1923)] = (val51+(f16(acc15))+val35);
}`;

const r_2048_16_40 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var val0 = data1[((gidx0*640)+(lidx0*40)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*0.0015625000232830644f)));
  }
}`;

const r_2048_16_40n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var val1 = data1[((gidx0*640)+(lidx0*40)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*0.0015625000232830644f)))+(f16(1e-05f)))));
  }
}`;

const E_64_10_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data2[alu1];
  var val1 = data3[alu1];
  var alu2 = (alu1+1);
  var val2 = data2[alu2];
  var val3 = data3[alu2];
  var alu3 = (alu1+2);
  var val4 = data2[alu3];
  var val5 = data3[alu3];
  var alu4 = (alu1+3);
  var val6 = data2[alu4];
  var val7 = data3[alu4];
  var alu5 = (lidx1<<2);
  var alu6 = (alu0+(gidx1*20480)+(lidx0*2560)+alu5);
  var val8 = data1[alu6];
  var alu7 = (alu6+1);
  var val9 = data1[alu7];
  var alu8 = (alu6+2);
  var val10 = data1[alu8];
  var alu9 = (alu6+3);
  var val11 = data1[alu9];
  var alu10 = (alu6+640);
  var val12 = data1[alu10];
  var alu11 = (alu6+641);
  var val13 = data1[alu11];
  var alu12 = (alu6+642);
  var val14 = data1[alu12];
  var alu13 = (alu6+643);
  var val15 = data1[alu13];
  var alu14 = (alu6+1280);
  var val16 = data1[alu14];
  var alu15 = (alu6+1281);
  var val17 = data1[alu15];
  var alu16 = (alu6+1282);
  var val18 = data1[alu16];
  var alu17 = (alu6+1283);
  var val19 = data1[alu17];
  var alu18 = (alu6+1920);
  var val20 = data1[alu18];
  var alu19 = (alu6+1921);
  var val21 = data1[alu19];
  var alu20 = (alu6+1922);
  var val22 = data1[alu20];
  var alu21 = (alu6+1923);
  var val23 = data1[alu21];
  var alu22 = (alu0+alu5);
  var val24 = data4[alu22];
  var val25 = data5[alu22];
  var alu23 = (alu22+1);
  var val26 = data4[alu23];
  var val27 = data5[alu23];
  var alu24 = (alu22+2);
  var val28 = data4[alu24];
  var val29 = data5[alu24];
  var alu25 = (alu22+3);
  var val30 = data4[alu25];
  var val31 = data5[alu25];
  data0[alu7] = (val27+(val26*val1*(val9-val0)));
  data0[alu8] = (val29+(val28*val1*(val10-val0)));
  data0[alu9] = (val31+(val30*val1*(val11-val0)));
  data0[alu10] = (val25+(val24*val3*(val12-val2)));
  data0[alu11] = (val27+(val26*val3*(val13-val2)));
  data0[alu12] = (val29+(val28*val3*(val14-val2)));
  data0[alu13] = (val31+(val30*val3*(val15-val2)));
  data0[alu14] = (val25+(val24*val5*(val16-val4)));
  data0[alu15] = (val27+(val26*val5*(val17-val4)));
  data0[alu16] = (val29+(val28*val5*(val18-val4)));
  data0[alu17] = (val31+(val30*val5*(val19-val4)));
  data0[alu18] = (val25+(val24*val7*(val20-val6)));
  data0[alu19] = (val27+(val26*val7*(val21-val6)));
  data0[alu20] = (val29+(val28*val7*(val22-val6)));
  data0[alu21] = (val31+(val30*val7*(val23-val6)));
  data0[alu6] = (val25+(val24*val1*(val8-val0)));
}`;

const r_2_16_77_8_16_20_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 16 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0*80);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 20; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx1*40960)+(gidx2*655360)+alu0+(lidx1*2560)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+640)];
    var val5 = data1[(alu2+641)];
    var val6 = data1[(alu2+642)];
    var val7 = data1[(alu2+643)];
    var val8 = data1[(alu2+1280)];
    var val9 = data1[(alu2+1281)];
    var val10 = data1[(alu2+1282)];
    var val11 = data1[(alu2+1283)];
    var val12 = data1[(alu2+1920)];
    var val13 = data1[(alu2+1921)];
    var val14 = data1[(alu2+1922)];
    var val15 = data1[(alu2+1923)];
    var alu3 = ((gidx0*640)+(gidx2*49280)+alu0+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
  }
  var alu9 = (gidx0+(gidx1*4928)+(gidx2*630784)+(lidx0*78848)+(lidx1*308));
  data0[alu9] = (acc0*0.11180339753627777f);
  data0[(alu9+77)] = (acc1*0.11180339753627777f);
  data0[(alu9+154)] = (acc2*0.11180339753627777f);
  data0[(alu9+231)] = (acc3*0.11180339753627777f);
}`;

const r_512_32_77 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val0 = data1[((gidx0*2464)+(lidx0*77)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_128_32_77_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu4 = ((gidx0*9856)+(lidx0*308)+ridx0);
    var val4 = data1[alu4];
    var val5 = data1[(alu4+77)];
    var val6 = data1[(alu4+154)];
    var val7 = data1[(alu4+231)];
    acc0 = (acc0+exp2(((val4-val3)*1.4426950408889634f)));
    acc1 = (acc1+exp2(((val5-val0)*1.4426950408889634f)));
    acc2 = (acc2+exp2(((val6-val1)*1.4426950408889634f)));
    acc3 = (acc3+exp2(((val7-val2)*1.4426950408889634f)));
  }
  data0[alu1] = acc1;
  data0[alu2] = acc2;
  data0[alu3] = acc3;
  data0[alu0] = acc0;
}`;

const E_128_77_32_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (gidx0+(gidx1*9856)+(lidx0*308));
  var val0 = data1[alu0];
  var alu1 = (alu0+77);
  var val1 = data1[alu1];
  var alu2 = (alu0+154);
  var val2 = data1[alu2];
  var alu3 = (alu0+231);
  var val3 = data1[alu3];
  var alu4 = ((gidx1<<7)+(lidx0<<2));
  var val4 = data2[alu4];
  var val5 = data3[alu4];
  var alu5 = (alu4+1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = (alu4+2);
  var val8 = data2[alu6];
  var val9 = data3[alu6];
  var alu7 = (alu4+3);
  var val10 = data2[alu7];
  var val11 = data3[alu7];
  data0[alu0] = (f16((exp2(((val0-val4)*1.4426950408889634f))*(1/val5))));
  data0[alu1] = (f16((exp2(((val1-val6)*1.4426950408889634f))*(1/val7))));
  data0[alu2] = (f16((exp2(((val2-val8)*1.4426950408889634f))*(1/val9))));
  data0[alu3] = (f16((exp2(((val3-val10)*1.4426950408889634f))*(1/val11))));
}`;

const r_2_4_16_5_2_16_4_77_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 80 */
  var gidx1 = i32(gindex.y); /* 4 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (gidx0/5);
  var alu1 = ((gidx0%5)<<4);
  var alu2 = (lidx2<<2);
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu3 = ((gidx1*157696)+(gidx2*630784)+(alu0*4928)+(lidx0*78848)+(lidx1*308)+ridx0);
    var val0 = data1[alu3];
    var val1 = data1[(alu3+77)];
    var val2 = data1[(alu3+154)];
    var val3 = data1[(alu3+231)];
    var alu4 = ((gidx1*160)+(gidx2*49280)+alu1+(lidx0*80)+alu2+(ridx0*640));
    var val4 = data2[alu4];
    var val5 = data2[(alu4+1)];
    var val6 = data2[(alu4+2)];
    var val7 = data2[(alu4+3)];
    acc0 = (acc0+(f32((val4*val0))));
    acc1 = (acc1+(f32((val4*val1))));
    acc2 = (acc2+(f32((val4*val2))));
    acc3 = (acc3+(f32((val4*val3))));
    acc4 = (acc4+(f32((val5*val0))));
    acc5 = (acc5+(f32((val5*val1))));
    acc6 = (acc6+(f32((val5*val2))));
    acc7 = (acc7+(f32((val5*val3))));
    acc8 = (acc8+(f32((val6*val0))));
    acc9 = (acc9+(f32((val6*val1))));
    acc10 = (acc10+(f32((val6*val2))));
    acc11 = (acc11+(f32((val6*val3))));
    acc12 = (acc12+(f32((val7*val0))));
    acc13 = (acc13+(f32((val7*val1))));
    acc14 = (acc14+(f32((val7*val2))));
    acc15 = (acc15+(f32((val7*val3))));
  }
  var alu22 = ((gidx1*163840)+(gidx2*655360)+(alu0*5120)+alu1+(lidx0*81920)+(lidx1*320)+alu2);
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+80)] = (f16(acc1));
  data0[(alu22+81)] = (f16(acc5));
  data0[(alu22+82)] = (f16(acc9));
  data0[(alu22+83)] = (f16(acc13));
  data0[(alu22+160)] = (f16(acc2));
  data0[(alu22+161)] = (f16(acc6));
  data0[(alu22+162)] = (f16(acc10));
  data0[(alu22+163)] = (f16(acc14));
  data0[(alu22+240)] = (f16(acc3));
  data0[(alu22+241)] = (f16(acc7));
  data0[(alu22+242)] = (f16(acc11));
  data0[(alu22+243)] = (f16(acc15));
}`;

const r_2_32_10_8_16_160_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*655360);
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu1 = ((gidx1*2560)+(lidx0*320)+alu0+((ridx0/20)*81920)+((ridx0%20)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+80)];
    var val5 = data2[(alu1+81)];
    var val6 = data2[(alu1+82)];
    var val7 = data2[(alu1+83)];
    var val8 = data2[(alu1+160)];
    var val9 = data2[(alu1+161)];
    var val10 = data2[(alu1+162)];
    var val11 = data2[(alu1+163)];
    var val12 = data2[(alu1+240)];
    var val13 = data2[(alu1+241)];
    var val14 = data2[(alu1+242)];
    var val15 = data2[(alu1+243)];
    var alu2 = ((gidx0*40960)+(lidx1*2560)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+640)];
    var val21 = data3[(alu2+641)];
    var val22 = data3[(alu2+642)];
    var val23 = data3[(alu2+643)];
    var val24 = data3[(alu2+1280)];
    var val25 = data3[(alu2+1281)];
    var val26 = data3[(alu2+1282)];
    var val27 = data3[(alu2+1283)];
    var val28 = data3[(alu2+1920)];
    var val29 = data3[(alu2+1921)];
    var val30 = data3[(alu2+1922)];
    var val31 = data3[(alu2+1923)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = ((gidx1*20480)+alu0+alu20+(lidx0*2560)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+640);
  var val36 = data1[alu26];
  var alu27 = (alu22+641);
  var val37 = data1[alu27];
  var alu28 = (alu22+642);
  var val38 = data1[alu28];
  var alu29 = (alu22+643);
  var val39 = data1[alu29];
  var alu30 = (alu22+1280);
  var val40 = data1[alu30];
  var alu31 = (alu22+1281);
  var val41 = data1[alu31];
  var alu32 = (alu22+1282);
  var val42 = data1[alu32];
  var alu33 = (alu22+1283);
  var val43 = data1[alu33];
  var alu34 = (alu22+1920);
  var val44 = data1[alu34];
  var alu35 = (alu22+1921);
  var val45 = data1[alu35];
  var alu36 = (alu22+1922);
  var val46 = data1[alu36];
  var alu37 = (alu22+1923);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_64_80_8_16_160_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 80 */
  var gidx1 = i32(gindex.y); /* 64 */
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*40960)+(lidx1*2560)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+640)];
    var val5 = data2[(alu1+641)];
    var val6 = data2[(alu1+642)];
    var val7 = data2[(alu1+643)];
    var val8 = data2[(alu1+1280)];
    var val9 = data2[(alu1+1281)];
    var val10 = data2[(alu1+1282)];
    var val11 = data2[(alu1+1283)];
    var val12 = data2[(alu1+1920)];
    var val13 = data2[(alu1+1921)];
    var val14 = data2[(alu1+1922)];
    var val15 = data2[(alu1+1923)];
    var alu2 = ((gidx1*20480)+(lidx0*2560)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+640)];
    var val21 = data1[(alu2+641)];
    var val22 = data1[(alu2+642)];
    var val23 = data1[(alu2+643)];
    var val24 = data1[(alu2+1280)];
    var val25 = data1[(alu2+1281)];
    var val26 = data1[(alu2+1282)];
    var val27 = data1[(alu2+1283)];
    var val28 = data1[(alu2+1920)];
    var val29 = data1[(alu2+1921)];
    var val30 = data1[(alu2+1922)];
    var val31 = data1[(alu2+1923)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = (alu20+(gidx1*163840)+(lidx0*20480)+alu21);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val33);
  data0[(alu23+2)] = ((f16(acc8))+val34);
  data0[(alu23+3)] = ((f16(acc12))+val35);
  data0[(alu23+5120)] = ((f16(acc1))+val32);
  data0[(alu23+5121)] = ((f16(acc5))+val33);
  data0[(alu23+5122)] = ((f16(acc9))+val34);
  data0[(alu23+5123)] = ((f16(acc13))+val35);
  data0[(alu23+10240)] = ((f16(acc2))+val32);
  data0[(alu23+10241)] = ((f16(acc6))+val33);
  data0[(alu23+10242)] = ((f16(acc10))+val34);
  data0[(alu23+10243)] = ((f16(acc14))+val35);
  data0[(alu23+15360)] = ((f16(acc3))+val32);
  data0[(alu23+15361)] = ((f16(acc7))+val33);
  data0[(alu23+15362)] = ((f16(acc11))+val34);
  data0[(alu23+15363)] = ((f16(acc15))+val35);
}`;

const E_256_40_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 256 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = (alu0+(gidx1*40960)+(lidx0*5120)+alu1);
  var val0 = data1[alu2];
  var val1 = data1[(alu2+1)];
  var val2 = data1[(alu2+2)];
  var val3 = data1[(alu2+3)];
  var val4 = data1[(alu2+2560)];
  var val5 = data1[(alu2+2561)];
  var val6 = data1[(alu2+2562)];
  var val7 = data1[(alu2+2563)];
  var alu3 = (alu0+(gidx1*20480)+(lidx0*2560)+alu1);
  data0[alu3] = (val0*(1/(exp2(((val4+(val4*val4*val4*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val4);
  data0[(alu3+1)] = (val1*(1/(exp2(((val5+(val5*val5*val5*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val5);
  data0[(alu3+2)] = (val2*(1/(exp2(((val6+(val6*val6*val6*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val6);
  data0[(alu3+3)] = (val3*(1/(exp2(((val7+(val7*val7*val7*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val7);
}`;

const r_64_10_8_16_640_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 64 */
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*163840)+(lidx1*10240)+alu0);
    var val0 = data3[alu1];
    var val1 = data3[(alu1+1)];
    var val2 = data3[(alu1+2)];
    var val3 = data3[(alu1+3)];
    var val4 = data3[(alu1+2560)];
    var val5 = data3[(alu1+2561)];
    var val6 = data3[(alu1+2562)];
    var val7 = data3[(alu1+2563)];
    var val8 = data3[(alu1+5120)];
    var val9 = data3[(alu1+5121)];
    var val10 = data3[(alu1+5122)];
    var val11 = data3[(alu1+5123)];
    var val12 = data3[(alu1+7680)];
    var val13 = data3[(alu1+7681)];
    var val14 = data3[(alu1+7682)];
    var val15 = data3[(alu1+7683)];
    var alu2 = ((gidx1*81920)+(lidx0*10240)+alu0);
    var val16 = data2[alu2];
    var val17 = data2[(alu2+1)];
    var val18 = data2[(alu2+2)];
    var val19 = data2[(alu2+3)];
    var val20 = data2[(alu2+2560)];
    var val21 = data2[(alu2+2561)];
    var val22 = data2[(alu2+2562)];
    var val23 = data2[(alu2+2563)];
    var val24 = data2[(alu2+5120)];
    var val25 = data2[(alu2+5121)];
    var val26 = data2[(alu2+5122)];
    var val27 = data2[(alu2+5123)];
    var val28 = data2[(alu2+7680)];
    var val29 = data2[(alu2+7681)];
    var val30 = data2[(alu2+7682)];
    var val31 = data2[(alu2+7683)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+(gidx1*20480)+(lidx0*2560)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+640);
  var val36 = data1[alu26];
  var alu27 = (alu22+641);
  var val37 = data1[alu27];
  var alu28 = (alu22+642);
  var val38 = data1[alu28];
  var alu29 = (alu22+643);
  var val39 = data1[alu29];
  var alu30 = (alu22+1280);
  var val40 = data1[alu30];
  var alu31 = (alu22+1281);
  var val41 = data1[alu31];
  var alu32 = (alu22+1282);
  var val42 = data1[alu32];
  var alu33 = (alu22+1283);
  var val43 = data1[alu33];
  var alu34 = (alu22+1920);
  var val44 = data1[alu34];
  var alu35 = (alu22+1921);
  var val45 = data1[alu35];
  var alu36 = (alu22+1922);
  var val46 = data1[alu36];
  var alu37 = (alu22+1923);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_2_20_16_8_16_160_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 20 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*655360);
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*40960)+alu0+(lidx1*2560)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+640)];
    var val5 = data1[(alu2+641)];
    var val6 = data1[(alu2+642)];
    var val7 = data1[(alu2+643)];
    var val8 = data1[(alu2+1280)];
    var val9 = data1[(alu2+1281)];
    var val10 = data1[(alu2+1282)];
    var val11 = data1[(alu2+1283)];
    var val12 = data1[(alu2+1920)];
    var val13 = data1[(alu2+1921)];
    var val14 = data1[(alu2+1922)];
    var val15 = data1[(alu2+1923)];
    var alu3 = ((gidx1*20480)+(lidx0*2560)+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    var val20 = data2[(alu3+640)];
    var val21 = data2[(alu3+641)];
    var val22 = data2[(alu3+642)];
    var val23 = data2[(alu3+643)];
    var val24 = data2[(alu3+1280)];
    var val25 = data2[(alu3+1281)];
    var val26 = data2[(alu3+1282)];
    var val27 = data2[(alu3+1283)];
    var val28 = data2[(alu3+1920)];
    var val29 = data2[(alu3+1921)];
    var val30 = data2[(alu3+1922)];
    var val31 = data2[(alu3+1923)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu21 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data3[(alu21+1)];
  var val34 = data3[(alu21+2)];
  var val35 = data3[(alu21+3)];
  var alu22 = ((gidx1<<15)+alu0+(gidx0<<6)+(lidx0<<12)+(lidx1<<2));
  var val36 = data4[alu22];
  var alu23 = (alu22+1);
  var val37 = data4[alu23];
  var alu24 = (alu22+2);
  var val38 = data4[alu24];
  var alu25 = (alu22+3);
  var val39 = data4[alu25];
  var alu26 = (alu22+1024);
  var val40 = data4[alu26];
  var alu27 = (alu22+1025);
  var val41 = data4[alu27];
  var alu28 = (alu22+1026);
  var val42 = data4[alu28];
  var alu29 = (alu22+1027);
  var val43 = data4[alu29];
  var alu30 = (alu22+2048);
  var val44 = data4[alu30];
  var alu31 = (alu22+2049);
  var val45 = data4[alu31];
  var alu32 = (alu22+2050);
  var val46 = data4[alu32];
  var alu33 = (alu22+2051);
  var val47 = data4[alu33];
  var alu34 = (alu22+3072);
  var val48 = data4[alu34];
  var alu35 = (alu22+3073);
  var val49 = data4[alu35];
  var alu36 = (alu22+3074);
  var val50 = data4[alu36];
  var alu37 = (alu22+3075);
  var val51 = data4[alu37];
  data0[alu23] = (val37+(f16(acc4))+val32);
  data0[alu24] = (val38+(f16(acc8))+val32);
  data0[alu25] = (val39+(f16(acc12))+val32);
  data0[alu26] = (val40+(f16(acc1))+val33);
  data0[alu27] = (val41+(f16(acc5))+val33);
  data0[alu28] = (val42+(f16(acc9))+val33);
  data0[alu29] = (val43+(f16(acc13))+val33);
  data0[alu30] = (val44+(f16(acc2))+val34);
  data0[alu31] = (val45+(f16(acc6))+val34);
  data0[alu32] = (val46+(f16(acc10))+val34);
  data0[alu33] = (val47+(f16(acc14))+val34);
  data0[alu34] = (val48+(f16(acc3))+val35);
  data0[alu35] = (val49+(f16(acc7))+val35);
  data0[alu36] = (val50+(f16(acc11))+val35);
  data0[alu37] = (val51+(f16(acc15))+val35);
  data0[alu22] = (val36+(f16(acc0))+val32);
}`;

const r_2_160_2_16_8_640_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx2*655360);
  var alu1 = (gidx0<<9);
  var alu2 = (lidx0<<5);
  var alu3 = (lidx1<<2);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (((gidx0+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx0<<4))<31);
  var alu7 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu8 = ((gidx1*23040)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+5760)];
    var val10 = data2[(alu8+5761)];
    var val11 = data2[(alu8+5762)];
    var val12 = data2[(alu8+5763)];
    var val13 = data2[(alu8+5764)];
    var val14 = data2[(alu8+5765)];
    var val15 = data2[(alu8+5766)];
    var val16 = data2[(alu8+5767)];
    var val17 = data2[(alu8+5768)];
    var val18 = data2[(alu8+11520)];
    var val19 = data2[(alu8+11521)];
    var val20 = data2[(alu8+11522)];
    var val21 = data2[(alu8+11523)];
    var val22 = data2[(alu8+11524)];
    var val23 = data2[(alu8+11525)];
    var val24 = data2[(alu8+11526)];
    var val25 = data2[(alu8+11527)];
    var val26 = data2[(alu8+11528)];
    var val27 = data2[(alu8+17280)];
    var val28 = data2[(alu8+17281)];
    var val29 = data2[(alu8+17282)];
    var val30 = data2[(alu8+17283)];
    var val31 = data2[(alu8+17284)];
    var val32 = data2[(alu8+17285)];
    var val33 = data2[(alu8+17286)];
    var val34 = data2[(alu8+17287)];
    var val35 = data2[(alu8+17288)];
    var alu9 = (alu1+alu2+alu0+(ridx0<<10)+alu3);
    var val36 = data1[alu9];
    var val37 = select((f16(0.0f)), data1[(alu9+-33)], (alu4&alu5));
    var val38 = select((f16(0.0f)), data1[(alu9+-32)], alu5);
    var val39 = select((f16(0.0f)), data1[(alu9+-31)], alu5);
    var val40 = select((f16(0.0f)), data1[(alu9+-30)], alu5);
    var val41 = select((f16(0.0f)), data1[(alu9+-29)], alu5);
    var val42 = select((f16(0.0f)), data1[(alu9+-28)], (alu7&alu5));
    var val43 = select((f16(0.0f)), data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select((f16(0.0f)), data1[(alu9+4)], alu7);
    var val48 = select((f16(0.0f)), data1[(alu9+31)], (alu6&alu4));
    var val49 = select((f16(0.0f)), data1[(alu9+32)], alu6);
    var val50 = select((f16(0.0f)), data1[(alu9+33)], alu6);
    var val51 = select((f16(0.0f)), data1[(alu9+34)], alu6);
    var val52 = select((f16(0.0f)), data1[(alu9+35)], alu6);
    var val53 = select((f16(0.0f)), data1[(alu9+36)], (alu7&alu6));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu27 = (gidx1<<2);
  var val54 = data3[alu27];
  var val55 = data4[alu27];
  var alu28 = (alu27+1);
  var val56 = data3[alu28];
  var val57 = data4[alu28];
  var alu29 = (alu27+2);
  var val58 = data3[alu29];
  var val59 = data4[alu29];
  var alu30 = (alu27+3);
  var val60 = data3[alu30];
  var val61 = data4[alu30];
  var alu31 = ((gidx1<<12)+alu0+alu1+alu2+alu3);
  data0[alu31] = (val55+(f16(acc0))+val54);
  data0[(alu31+1)] = (val55+(f16(acc4))+val54);
  data0[(alu31+2)] = (val55+(f16(acc8))+val54);
  data0[(alu31+3)] = (val55+(f16(acc12))+val54);
  data0[(alu31+1024)] = (val57+(f16(acc1))+val56);
  data0[(alu31+1025)] = (val57+(f16(acc5))+val56);
  data0[(alu31+1026)] = (val57+(f16(acc9))+val56);
  data0[(alu31+1027)] = (val57+(f16(acc13))+val56);
  data0[(alu31+2048)] = (val59+(f16(acc2))+val58);
  data0[(alu31+2049)] = (val59+(f16(acc6))+val58);
  data0[(alu31+2050)] = (val59+(f16(acc10))+val58);
  data0[(alu31+2051)] = (val59+(f16(acc14))+val58);
  data0[(alu31+3072)] = (val61+(f16(acc3))+val60);
  data0[(alu31+3073)] = (val61+(f16(acc7))+val60);
  data0[(alu31+3074)] = (val61+(f16(acc11))+val60);
  data0[(alu31+3075)] = (val61+(f16(acc15))+val60);
}`;

const r_2_160_2_16_8_640_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx2*655360);
  var alu1 = (gidx0<<9);
  var alu2 = (lidx0<<5);
  var alu3 = (lidx1<<2);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (((gidx0+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx0<<4))<31);
  var alu7 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu8 = ((gidx1*23040)+(ridx0*9));
    var val0 = data3[alu8];
    var val1 = data3[(alu8+1)];
    var val2 = data3[(alu8+2)];
    var val3 = data3[(alu8+3)];
    var val4 = data3[(alu8+4)];
    var val5 = data3[(alu8+5)];
    var val6 = data3[(alu8+6)];
    var val7 = data3[(alu8+7)];
    var val8 = data3[(alu8+8)];
    var val9 = data3[(alu8+5760)];
    var val10 = data3[(alu8+5761)];
    var val11 = data3[(alu8+5762)];
    var val12 = data3[(alu8+5763)];
    var val13 = data3[(alu8+5764)];
    var val14 = data3[(alu8+5765)];
    var val15 = data3[(alu8+5766)];
    var val16 = data3[(alu8+5767)];
    var val17 = data3[(alu8+5768)];
    var val18 = data3[(alu8+11520)];
    var val19 = data3[(alu8+11521)];
    var val20 = data3[(alu8+11522)];
    var val21 = data3[(alu8+11523)];
    var val22 = data3[(alu8+11524)];
    var val23 = data3[(alu8+11525)];
    var val24 = data3[(alu8+11526)];
    var val25 = data3[(alu8+11527)];
    var val26 = data3[(alu8+11528)];
    var val27 = data3[(alu8+17280)];
    var val28 = data3[(alu8+17281)];
    var val29 = data3[(alu8+17282)];
    var val30 = data3[(alu8+17283)];
    var val31 = data3[(alu8+17284)];
    var val32 = data3[(alu8+17285)];
    var val33 = data3[(alu8+17286)];
    var val34 = data3[(alu8+17287)];
    var val35 = data3[(alu8+17288)];
    var alu9 = (alu1+alu2+alu0+(ridx0<<10)+alu3);
    var val36 = data2[alu9];
    var val37 = select((f16(0.0f)), data2[(alu9+-33)], (alu4&alu5));
    var val38 = select((f16(0.0f)), data2[(alu9+-32)], alu5);
    var val39 = select((f16(0.0f)), data2[(alu9+-31)], alu5);
    var val40 = select((f16(0.0f)), data2[(alu9+-30)], alu5);
    var val41 = select((f16(0.0f)), data2[(alu9+-29)], alu5);
    var val42 = select((f16(0.0f)), data2[(alu9+-28)], (alu7&alu5));
    var val43 = select((f16(0.0f)), data2[(alu9+-1)], alu4);
    var val44 = data2[(alu9+1)];
    var val45 = data2[(alu9+2)];
    var val46 = data2[(alu9+3)];
    var val47 = select((f16(0.0f)), data2[(alu9+4)], alu7);
    var val48 = select((f16(0.0f)), data2[(alu9+31)], (alu6&alu4));
    var val49 = select((f16(0.0f)), data2[(alu9+32)], alu6);
    var val50 = select((f16(0.0f)), data2[(alu9+33)], alu6);
    var val51 = select((f16(0.0f)), data2[(alu9+34)], alu6);
    var val52 = select((f16(0.0f)), data2[(alu9+35)], alu6);
    var val53 = select((f16(0.0f)), data2[(alu9+36)], (alu7&alu6));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu27 = (gidx1<<2);
  var val54 = data4[alu27];
  var val55 = data4[(alu27+1)];
  var val56 = data4[(alu27+2)];
  var val57 = data4[(alu27+3)];
  var alu28 = ((gidx1<<12)+alu0+alu1+alu2+alu3);
  var val58 = data1[alu28];
  var alu29 = (alu28+1);
  var val59 = data1[alu29];
  var alu30 = (alu28+2);
  var val60 = data1[alu30];
  var alu31 = (alu28+3);
  var val61 = data1[alu31];
  var alu32 = (alu28+1024);
  var val62 = data1[alu32];
  var alu33 = (alu28+1025);
  var val63 = data1[alu33];
  var alu34 = (alu28+1026);
  var val64 = data1[alu34];
  var alu35 = (alu28+1027);
  var val65 = data1[alu35];
  var alu36 = (alu28+2048);
  var val66 = data1[alu36];
  var alu37 = (alu28+2049);
  var val67 = data1[alu37];
  var alu38 = (alu28+2050);
  var val68 = data1[alu38];
  var alu39 = (alu28+2051);
  var val69 = data1[alu39];
  var alu40 = (alu28+3072);
  var val70 = data1[alu40];
  var alu41 = (alu28+3073);
  var val71 = data1[alu41];
  var alu42 = (alu28+3074);
  var val72 = data1[alu42];
  var alu43 = (alu28+3075);
  var val73 = data1[alu43];
  data0[alu29] = (val59+(f16(acc4))+val54);
  data0[alu30] = (val60+(f16(acc8))+val54);
  data0[alu31] = (val61+(f16(acc12))+val54);
  data0[alu32] = (val62+(f16(acc1))+val55);
  data0[alu33] = (val63+(f16(acc5))+val55);
  data0[alu34] = (val64+(f16(acc9))+val55);
  data0[alu35] = (val65+(f16(acc13))+val55);
  data0[alu36] = (val66+(f16(acc2))+val56);
  data0[alu37] = (val67+(f16(acc6))+val56);
  data0[alu38] = (val68+(f16(acc10))+val56);
  data0[alu39] = (val69+(f16(acc14))+val56);
  data0[alu40] = (val70+(f16(acc3))+val57);
  data0[alu41] = (val71+(f16(acc7))+val57);
  data0[alu42] = (val72+(f16(acc11))+val57);
  data0[alu43] = (val73+(f16(acc15))+val57);
  data0[alu28] = (val58+(f16(acc0))+val54);
}`;

const r_2_80_2_16_4_640_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 80 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = ((lidx1<1)!=true);
  var alu1 = ((lidx2<1)!=true);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu2 = ((gidx0*46080)+(lidx0*23040)+(ridx0*9));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+4)];
    var val5 = data2[(alu2+5)];
    var val6 = data2[(alu2+6)];
    var val7 = data2[(alu2+7)];
    var val8 = data2[(alu2+8)];
    var val9 = data2[(alu2+5760)];
    var val10 = data2[(alu2+5761)];
    var val11 = data2[(alu2+5762)];
    var val12 = data2[(alu2+5763)];
    var val13 = data2[(alu2+5764)];
    var val14 = data2[(alu2+5765)];
    var val15 = data2[(alu2+5766)];
    var val16 = data2[(alu2+5767)];
    var val17 = data2[(alu2+5768)];
    var val18 = data2[(alu2+11520)];
    var val19 = data2[(alu2+11521)];
    var val20 = data2[(alu2+11522)];
    var val21 = data2[(alu2+11523)];
    var val22 = data2[(alu2+11524)];
    var val23 = data2[(alu2+11525)];
    var val24 = data2[(alu2+11526)];
    var val25 = data2[(alu2+11527)];
    var val26 = data2[(alu2+11528)];
    var val27 = data2[(alu2+17280)];
    var val28 = data2[(alu2+17281)];
    var val29 = data2[(alu2+17282)];
    var val30 = data2[(alu2+17283)];
    var val31 = data2[(alu2+17284)];
    var val32 = data2[(alu2+17285)];
    var val33 = data2[(alu2+17286)];
    var val34 = data2[(alu2+17287)];
    var val35 = data2[(alu2+17288)];
    var alu3 = ((gidx1*655360)+(ridx0<<10)+(lidx1<<6)+(lidx2<<3));
    var val36 = data1[alu3];
    var val37 = select((f16(0.0f)), data1[(alu3+-33)], (alu0&alu1));
    var val38 = select((f16(0.0f)), data1[(alu3+-32)], alu0);
    var val39 = select((f16(0.0f)), data1[(alu3+-31)], alu0);
    var val40 = select((f16(0.0f)), data1[(alu3+-30)], alu0);
    var val41 = select((f16(0.0f)), data1[(alu3+-29)], alu0);
    var val42 = select((f16(0.0f)), data1[(alu3+-28)], alu0);
    var val43 = select((f16(0.0f)), data1[(alu3+-27)], alu0);
    var val44 = select((f16(0.0f)), data1[(alu3+-26)], alu0);
    var val45 = select((f16(0.0f)), data1[(alu3+-25)], alu0);
    var val46 = select((f16(0.0f)), data1[(alu3+-1)], alu1);
    var val47 = data1[(alu3+1)];
    var val48 = data1[(alu3+2)];
    var val49 = data1[(alu3+3)];
    var val50 = data1[(alu3+4)];
    var val51 = data1[(alu3+5)];
    var val52 = data1[(alu3+6)];
    var val53 = data1[(alu3+7)];
    var val54 = select((f16(0.0f)), data1[(alu3+31)], alu1);
    var val55 = data1[(alu3+32)];
    var val56 = data1[(alu3+33)];
    var val57 = data1[(alu3+34)];
    var val58 = data1[(alu3+35)];
    var val59 = data1[(alu3+36)];
    var val60 = data1[(alu3+37)];
    var val61 = data1[(alu3+38)];
    var val62 = data1[(alu3+39)];
    acc0 = (acc0+(f32((val56*val8)))+(f32((val47*val5)))+(f32((val39*val2)))+(f32((val55*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val54*val6)))+(f32((val37*val0)))+(f32((val46*val3))));
    acc1 = (acc1+(f32((val56*val17)))+(f32((val47*val14)))+(f32((val39*val11)))+(f32((val55*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val54*val15)))+(f32((val37*val9)))+(f32((val46*val12))));
    acc2 = (acc2+(f32((val56*val26)))+(f32((val47*val23)))+(f32((val39*val20)))+(f32((val55*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val54*val24)))+(f32((val37*val18)))+(f32((val46*val21))));
    acc3 = (acc3+(f32((val56*val35)))+(f32((val47*val32)))+(f32((val39*val29)))+(f32((val55*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val54*val33)))+(f32((val37*val27)))+(f32((val46*val30))));
    acc4 = (acc4+(f32((val58*val8)))+(f32((val49*val5)))+(f32((val41*val2)))+(f32((val57*val7)))+(f32((val48*val4)))+(f32((val40*val1)))+(f32((val56*val6)))+(f32((val39*val0)))+(f32((val47*val3))));
    acc5 = (acc5+(f32((val58*val17)))+(f32((val49*val14)))+(f32((val41*val11)))+(f32((val57*val16)))+(f32((val48*val13)))+(f32((val40*val10)))+(f32((val56*val15)))+(f32((val39*val9)))+(f32((val47*val12))));
    acc6 = (acc6+(f32((val58*val26)))+(f32((val49*val23)))+(f32((val41*val20)))+(f32((val57*val25)))+(f32((val48*val22)))+(f32((val40*val19)))+(f32((val56*val24)))+(f32((val39*val18)))+(f32((val47*val21))));
    acc7 = (acc7+(f32((val58*val35)))+(f32((val49*val32)))+(f32((val41*val29)))+(f32((val57*val34)))+(f32((val48*val31)))+(f32((val40*val28)))+(f32((val56*val33)))+(f32((val39*val27)))+(f32((val47*val30))));
    acc8 = (acc8+(f32((val60*val8)))+(f32((val51*val5)))+(f32((val43*val2)))+(f32((val59*val7)))+(f32((val50*val4)))+(f32((val42*val1)))+(f32((val58*val6)))+(f32((val41*val0)))+(f32((val49*val3))));
    acc9 = (acc9+(f32((val60*val17)))+(f32((val51*val14)))+(f32((val43*val11)))+(f32((val59*val16)))+(f32((val50*val13)))+(f32((val42*val10)))+(f32((val58*val15)))+(f32((val41*val9)))+(f32((val49*val12))));
    acc10 = (acc10+(f32((val60*val26)))+(f32((val51*val23)))+(f32((val43*val20)))+(f32((val59*val25)))+(f32((val50*val22)))+(f32((val42*val19)))+(f32((val58*val24)))+(f32((val41*val18)))+(f32((val49*val21))));
    acc11 = (acc11+(f32((val60*val35)))+(f32((val51*val32)))+(f32((val43*val29)))+(f32((val59*val34)))+(f32((val50*val31)))+(f32((val42*val28)))+(f32((val58*val33)))+(f32((val41*val27)))+(f32((val49*val30))));
    acc12 = (acc12+(f32((val62*val8)))+(f32((val53*val5)))+(f32((val45*val2)))+(f32((val61*val7)))+(f32((val52*val4)))+(f32((val44*val1)))+(f32((val60*val6)))+(f32((val43*val0)))+(f32((val51*val3))));
    acc13 = (acc13+(f32((val62*val17)))+(f32((val53*val14)))+(f32((val45*val11)))+(f32((val61*val16)))+(f32((val52*val13)))+(f32((val44*val10)))+(f32((val60*val15)))+(f32((val43*val9)))+(f32((val51*val12))));
    acc14 = (acc14+(f32((val62*val26)))+(f32((val53*val23)))+(f32((val45*val20)))+(f32((val61*val25)))+(f32((val52*val22)))+(f32((val44*val19)))+(f32((val60*val24)))+(f32((val43*val18)))+(f32((val51*val21))));
    acc15 = (acc15+(f32((val62*val35)))+(f32((val53*val32)))+(f32((val45*val29)))+(f32((val61*val34)))+(f32((val52*val31)))+(f32((val44*val28)))+(f32((val60*val33)))+(f32((val43*val27)))+(f32((val51*val30))));
  }
  var alu21 = ((gidx0<<3)+(lidx0<<2));
  var val63 = data3[alu21];
  var val64 = data3[(alu21+1)];
  var val65 = data3[(alu21+2)];
  var val66 = data3[(alu21+3)];
  var alu22 = ((gidx0<<11)+(gidx1*163840)+(lidx0<<10)+(lidx1<<4)+(lidx2<<2));
  data0[alu22] = ((f16(acc0))+val63);
  data0[(alu22+1)] = ((f16(acc4))+val63);
  data0[(alu22+2)] = ((f16(acc8))+val63);
  data0[(alu22+3)] = ((f16(acc12))+val63);
  data0[(alu22+256)] = ((f16(acc1))+val64);
  data0[(alu22+257)] = ((f16(acc5))+val64);
  data0[(alu22+258)] = ((f16(acc9))+val64);
  data0[(alu22+259)] = ((f16(acc13))+val64);
  data0[(alu22+512)] = ((f16(acc2))+val65);
  data0[(alu22+513)] = ((f16(acc6))+val65);
  data0[(alu22+514)] = ((f16(acc10))+val65);
  data0[(alu22+515)] = ((f16(acc14))+val65);
  data0[(alu22+768)] = ((f16(acc3))+val66);
  data0[(alu22+769)] = ((f16(acc7))+val66);
  data0[(alu22+770)] = ((f16(acc11))+val66);
  data0[(alu22+771)] = ((f16(acc15))+val66);
}`;

const r_64_16_320 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var val0 = data1[((gidx0*5120)+(lidx0*320)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*0.00019531250291038305f)));
  }
}`;

const r_64_16_320n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var val1 = data1[((gidx0*5120)+(lidx0*320)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*0.00019531250291038305f)))+(f16(1e-05f)))));
  }
}`;

const E_2_80_4_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/20));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<11)+(gidx2*163840)+(gidx0<<6)+(lidx0<<8)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_4_640_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (lidx1<<4);
  var alu1 = (lidx2<<2);
  var alu2 = ((lidx1<1)!=true);
  var alu3 = ((lidx2<1)!=true);
  var alu4 = (lidx1<15);
  var alu5 = (lidx2<3);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu6 = ((gidx0*46080)+(lidx0*23040)+(ridx0*9));
    var val0 = data2[alu6];
    var val1 = data2[(alu6+1)];
    var val2 = data2[(alu6+2)];
    var val3 = data2[(alu6+3)];
    var val4 = data2[(alu6+4)];
    var val5 = data2[(alu6+5)];
    var val6 = data2[(alu6+6)];
    var val7 = data2[(alu6+7)];
    var val8 = data2[(alu6+8)];
    var val9 = data2[(alu6+5760)];
    var val10 = data2[(alu6+5761)];
    var val11 = data2[(alu6+5762)];
    var val12 = data2[(alu6+5763)];
    var val13 = data2[(alu6+5764)];
    var val14 = data2[(alu6+5765)];
    var val15 = data2[(alu6+5766)];
    var val16 = data2[(alu6+5767)];
    var val17 = data2[(alu6+5768)];
    var val18 = data2[(alu6+11520)];
    var val19 = data2[(alu6+11521)];
    var val20 = data2[(alu6+11522)];
    var val21 = data2[(alu6+11523)];
    var val22 = data2[(alu6+11524)];
    var val23 = data2[(alu6+11525)];
    var val24 = data2[(alu6+11526)];
    var val25 = data2[(alu6+11527)];
    var val26 = data2[(alu6+11528)];
    var val27 = data2[(alu6+17280)];
    var val28 = data2[(alu6+17281)];
    var val29 = data2[(alu6+17282)];
    var val30 = data2[(alu6+17283)];
    var val31 = data2[(alu6+17284)];
    var val32 = data2[(alu6+17285)];
    var val33 = data2[(alu6+17286)];
    var val34 = data2[(alu6+17287)];
    var val35 = data2[(alu6+17288)];
    var alu7 = ((gidx1*163840)+(ridx0<<8)+alu0+alu1);
    var val36 = data1[alu7];
    var val37 = select((f16(0.0f)), data1[(alu7+-17)], (alu2&alu3));
    var val38 = select((f16(0.0f)), data1[(alu7+-16)], alu2);
    var val39 = select((f16(0.0f)), data1[(alu7+-15)], alu2);
    var val40 = select((f16(0.0f)), data1[(alu7+-14)], alu2);
    var val41 = select((f16(0.0f)), data1[(alu7+-13)], alu2);
    var val42 = select((f16(0.0f)), data1[(alu7+-12)], (alu5&alu2));
    var val43 = select((f16(0.0f)), data1[(alu7+-1)], alu3);
    var val44 = data1[(alu7+1)];
    var val45 = data1[(alu7+2)];
    var val46 = data1[(alu7+3)];
    var val47 = select((f16(0.0f)), data1[(alu7+4)], alu5);
    var val48 = select((f16(0.0f)), data1[(alu7+15)], (alu4&alu3));
    var val49 = select((f16(0.0f)), data1[(alu7+16)], alu4);
    var val50 = select((f16(0.0f)), data1[(alu7+17)], alu4);
    var val51 = select((f16(0.0f)), data1[(alu7+18)], alu4);
    var val52 = select((f16(0.0f)), data1[(alu7+19)], alu4);
    var val53 = select((f16(0.0f)), data1[(alu7+20)], (alu4&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu25 = ((gidx0<<3)+(lidx0<<2));
  var val54 = data3[alu25];
  var val55 = data4[alu25];
  var alu26 = (alu25+1);
  var val56 = data3[alu26];
  var val57 = data4[alu26];
  var alu27 = (alu25+2);
  var val58 = data3[alu27];
  var val59 = data4[alu27];
  var alu28 = (alu25+3);
  var val60 = data3[alu28];
  var val61 = data4[alu28];
  var alu29 = ((gidx0<<11)+(gidx1*327680)+(lidx0<<10)+alu0+alu1);
  data0[alu29] = (val55+(f16(acc0))+val54);
  data0[(alu29+1)] = (val55+(f16(acc4))+val54);
  data0[(alu29+2)] = (val55+(f16(acc8))+val54);
  data0[(alu29+3)] = (val55+(f16(acc12))+val54);
  data0[(alu29+256)] = (val57+(f16(acc1))+val56);
  data0[(alu29+257)] = (val57+(f16(acc5))+val56);
  data0[(alu29+258)] = (val57+(f16(acc9))+val56);
  data0[(alu29+259)] = (val57+(f16(acc13))+val56);
  data0[(alu29+512)] = (val59+(f16(acc2))+val58);
  data0[(alu29+513)] = (val59+(f16(acc6))+val58);
  data0[(alu29+514)] = (val59+(f16(acc10))+val58);
  data0[(alu29+515)] = (val59+(f16(acc14))+val58);
  data0[(alu29+768)] = (val61+(f16(acc3))+val60);
  data0[(alu29+769)] = (val61+(f16(acc7))+val60);
  data0[(alu29+770)] = (val61+(f16(acc11))+val60);
  data0[(alu29+771)] = (val61+(f16(acc15))+val60);
}`;

const E_2_160_4_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(gidx1/5));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<11)+(gidx2*327680)+(gidx0<<6)+(lidx0<<8)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_4_1280_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (gidx1*327680);
  var alu1 = (lidx1<<4);
  var alu2 = (lidx2<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = ((lidx2<1)!=true);
  var alu5 = (lidx1<15);
  var alu6 = (lidx2<3);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx0*92160)+(lidx0*46080)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+11520)];
    var val10 = data2[(alu7+11521)];
    var val11 = data2[(alu7+11522)];
    var val12 = data2[(alu7+11523)];
    var val13 = data2[(alu7+11524)];
    var val14 = data2[(alu7+11525)];
    var val15 = data2[(alu7+11526)];
    var val16 = data2[(alu7+11527)];
    var val17 = data2[(alu7+11528)];
    var val18 = data2[(alu7+23040)];
    var val19 = data2[(alu7+23041)];
    var val20 = data2[(alu7+23042)];
    var val21 = data2[(alu7+23043)];
    var val22 = data2[(alu7+23044)];
    var val23 = data2[(alu7+23045)];
    var val24 = data2[(alu7+23046)];
    var val25 = data2[(alu7+23047)];
    var val26 = data2[(alu7+23048)];
    var val27 = data2[(alu7+34560)];
    var val28 = data2[(alu7+34561)];
    var val29 = data2[(alu7+34562)];
    var val30 = data2[(alu7+34563)];
    var val31 = data2[(alu7+34564)];
    var val32 = data2[(alu7+34565)];
    var val33 = data2[(alu7+34566)];
    var val34 = data2[(alu7+34567)];
    var val35 = data2[(alu7+34568)];
    var alu8 = (alu0+(ridx0<<8)+alu1+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-17)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-16)], alu3);
    var val39 = select((f16(0.0f)), data1[(alu8+-15)], alu3);
    var val40 = select((f16(0.0f)), data1[(alu8+-14)], alu3);
    var val41 = select((f16(0.0f)), data1[(alu8+-13)], alu3);
    var val42 = select((f16(0.0f)), data1[(alu8+-12)], (alu6&alu3));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu4);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+15)], (alu5&alu4));
    var val49 = select((f16(0.0f)), data1[(alu8+16)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+17)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+18)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+19)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+20)], (alu5&alu6));
    acc0 = (acc0+(f32((val8*val50)))+(f32((val5*val44)))+(f32((val2*val39)))+(f32((val7*val49)))+(f32((val4*val36)))+(f32((val1*val38)))+(f32((val6*val48)))+(f32((val3*val43)))+(f32((val0*val37))));
    acc1 = (acc1+(f32((val8*val51)))+(f32((val5*val45)))+(f32((val2*val40)))+(f32((val7*val50)))+(f32((val4*val44)))+(f32((val1*val39)))+(f32((val6*val49)))+(f32((val3*val36)))+(f32((val0*val38))));
    acc2 = (acc2+(f32((val8*val52)))+(f32((val5*val46)))+(f32((val2*val41)))+(f32((val7*val51)))+(f32((val4*val45)))+(f32((val1*val40)))+(f32((val6*val50)))+(f32((val3*val44)))+(f32((val0*val39))));
    acc3 = (acc3+(f32((val8*val53)))+(f32((val5*val47)))+(f32((val2*val42)))+(f32((val7*val52)))+(f32((val4*val46)))+(f32((val1*val41)))+(f32((val6*val51)))+(f32((val3*val45)))+(f32((val0*val40))));
    acc4 = (acc4+(f32((val17*val50)))+(f32((val14*val44)))+(f32((val11*val39)))+(f32((val16*val49)))+(f32((val13*val36)))+(f32((val10*val38)))+(f32((val15*val48)))+(f32((val9*val37)))+(f32((val12*val43))));
    acc5 = (acc5+(f32((val17*val51)))+(f32((val14*val45)))+(f32((val11*val40)))+(f32((val16*val50)))+(f32((val13*val44)))+(f32((val10*val39)))+(f32((val15*val49)))+(f32((val9*val38)))+(f32((val12*val36))));
    acc6 = (acc6+(f32((val17*val52)))+(f32((val14*val46)))+(f32((val11*val41)))+(f32((val16*val51)))+(f32((val13*val45)))+(f32((val10*val40)))+(f32((val15*val50)))+(f32((val9*val39)))+(f32((val12*val44))));
    acc7 = (acc7+(f32((val17*val53)))+(f32((val14*val47)))+(f32((val11*val42)))+(f32((val16*val52)))+(f32((val13*val46)))+(f32((val10*val41)))+(f32((val15*val51)))+(f32((val9*val40)))+(f32((val12*val45))));
    acc8 = (acc8+(f32((val26*val50)))+(f32((val23*val44)))+(f32((val20*val39)))+(f32((val25*val49)))+(f32((val22*val36)))+(f32((val19*val38)))+(f32((val24*val48)))+(f32((val18*val37)))+(f32((val21*val43))));
    acc9 = (acc9+(f32((val26*val51)))+(f32((val23*val45)))+(f32((val20*val40)))+(f32((val25*val50)))+(f32((val22*val44)))+(f32((val19*val39)))+(f32((val24*val49)))+(f32((val18*val38)))+(f32((val21*val36))));
    acc10 = (acc10+(f32((val26*val52)))+(f32((val23*val46)))+(f32((val20*val41)))+(f32((val25*val51)))+(f32((val22*val45)))+(f32((val19*val40)))+(f32((val24*val50)))+(f32((val18*val39)))+(f32((val21*val44))));
    acc11 = (acc11+(f32((val26*val53)))+(f32((val23*val47)))+(f32((val20*val42)))+(f32((val25*val52)))+(f32((val22*val46)))+(f32((val19*val41)))+(f32((val24*val51)))+(f32((val18*val40)))+(f32((val21*val45))));
    acc12 = (acc12+(f32((val35*val50)))+(f32((val32*val44)))+(f32((val29*val39)))+(f32((val34*val49)))+(f32((val31*val36)))+(f32((val28*val38)))+(f32((val33*val48)))+(f32((val27*val37)))+(f32((val30*val43))));
    acc13 = (acc13+(f32((val35*val51)))+(f32((val32*val45)))+(f32((val29*val40)))+(f32((val34*val50)))+(f32((val31*val44)))+(f32((val28*val39)))+(f32((val33*val49)))+(f32((val27*val38)))+(f32((val30*val36))));
    acc14 = (acc14+(f32((val35*val52)))+(f32((val32*val46)))+(f32((val29*val41)))+(f32((val34*val51)))+(f32((val31*val45)))+(f32((val28*val40)))+(f32((val33*val50)))+(f32((val27*val39)))+(f32((val30*val44))));
    acc15 = (acc15+(f32((val35*val53)))+(f32((val32*val47)))+(f32((val29*val42)))+(f32((val34*val52)))+(f32((val31*val46)))+(f32((val28*val41)))+(f32((val33*val51)))+(f32((val27*val40)))+(f32((val30*val45))));
  }
  var alu26 = ((gidx0<<11)+alu0+(lidx0<<10)+alu1+alu2);
  data0[alu26] = acc0;
  data0[(alu26+1)] = acc1;
  data0[(alu26+2)] = acc2;
  data0[(alu26+3)] = acc3;
  data0[(alu26+256)] = acc4;
  data0[(alu26+257)] = acc5;
  data0[(alu26+258)] = acc6;
  data0[(alu26+259)] = acc7;
  data0[(alu26+512)] = acc8;
  data0[(alu26+513)] = acc9;
  data0[(alu26+514)] = acc10;
  data0[(alu26+515)] = acc11;
  data0[(alu26+768)] = acc12;
  data0[(alu26+769)] = acc13;
  data0[(alu26+770)] = acc14;
  data0[(alu26+771)] = acc15;
}`;

const r_2_40_4_8_16_160_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu2 = ((gidx1*20480)+(lidx0*2560)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+640)];
    var val5 = data2[(alu2+641)];
    var val6 = data2[(alu2+642)];
    var val7 = data2[(alu2+643)];
    var val8 = data2[(alu2+1280)];
    var val9 = data2[(alu2+1281)];
    var val10 = data2[(alu2+1282)];
    var val11 = data2[(alu2+1283)];
    var val12 = data2[(alu2+1920)];
    var val13 = data2[(alu2+1921)];
    var val14 = data2[(alu2+1922)];
    var val15 = data2[(alu2+1923)];
    var alu3 = (alu0+(gidx2*163840)+alu1+(ridx0<<10));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+256)];
    var val21 = data1[(alu3+257)];
    var val22 = data1[(alu3+258)];
    var val23 = data1[(alu3+259)];
    var val24 = data1[(alu3+512)];
    var val25 = data1[(alu3+513)];
    var val26 = data1[(alu3+514)];
    var val27 = data1[(alu3+515)];
    var val28 = data1[(alu3+768)];
    var val29 = data1[(alu3+769)];
    var val30 = data1[(alu3+770)];
    var val31 = data1[(alu3+771)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<13)+(gidx2*327680)+alu0+(lidx0<<10)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+256);
  var val44 = data4[alu29];
  var alu30 = (alu25+257);
  var val45 = data4[alu30];
  var alu31 = (alu25+258);
  var val46 = data4[alu31];
  var alu32 = (alu25+259);
  var val47 = data4[alu32];
  var alu33 = (alu25+512);
  var val48 = data4[alu33];
  var alu34 = (alu25+513);
  var val49 = data4[alu34];
  var alu35 = (alu25+514);
  var val50 = data4[alu35];
  var alu36 = (alu25+515);
  var val51 = data4[alu36];
  var alu37 = (alu25+768);
  var val52 = data4[alu37];
  var alu38 = (alu25+769);
  var val53 = data4[alu38];
  var alu39 = (alu25+770);
  var val54 = data4[alu39];
  var alu40 = (alu25+771);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const E_2_160_4_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(gidx1/5));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<11)+(gidx2*327680)+(gidx0<<6)+(lidx0<<8)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  data0[alu3] = (val1+(val0*val3*(val5-val2)));
  data0[alu4] = (val1+(val0*val3*(val6-val2)));
  data0[alu5] = (val1+(val0*val3*(val7-val2)));
  data0[alu2] = (val1+(val0*val3*(val4-val2)));
}`;

const r_2_40_4_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*327680);
  var alu1 = (gidx0<<6);
  var alu2 = (lidx1<<2);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu3 = ((gidx1*40960)+(lidx0*5120)+(ridx0<<2));
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+1280)];
    var val5 = data2[(alu3+1281)];
    var val6 = data2[(alu3+1282)];
    var val7 = data2[(alu3+1283)];
    var val8 = data2[(alu3+2560)];
    var val9 = data2[(alu3+2561)];
    var val10 = data2[(alu3+2562)];
    var val11 = data2[(alu3+2563)];
    var val12 = data2[(alu3+3840)];
    var val13 = data2[(alu3+3841)];
    var val14 = data2[(alu3+3842)];
    var val15 = data2[(alu3+3843)];
    var alu4 = (alu1+alu0+alu2+(ridx0<<10));
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+256)];
    var val21 = data1[(alu4+257)];
    var val22 = data1[(alu4+258)];
    var val23 = data1[(alu4+259)];
    var val24 = data1[(alu4+512)];
    var val25 = data1[(alu4+513)];
    var val26 = data1[(alu4+514)];
    var val27 = data1[(alu4+515)];
    var val28 = data1[(alu4+768)];
    var val29 = data1[(alu4+769)];
    var val30 = data1[(alu4+770)];
    var val31 = data1[(alu4+771)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
  }
  var alu22 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = ((gidx1<<13)+alu0+alu1+(lidx0<<10)+alu2);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val32);
  data0[(alu23+2)] = ((f16(acc8))+val32);
  data0[(alu23+3)] = ((f16(acc12))+val32);
  data0[(alu23+256)] = ((f16(acc1))+val33);
  data0[(alu23+257)] = ((f16(acc5))+val33);
  data0[(alu23+258)] = ((f16(acc9))+val33);
  data0[(alu23+259)] = ((f16(acc13))+val33);
  data0[(alu23+512)] = ((f16(acc2))+val34);
  data0[(alu23+513)] = ((f16(acc6))+val34);
  data0[(alu23+514)] = ((f16(acc10))+val34);
  data0[(alu23+515)] = ((f16(acc14))+val34);
  data0[(alu23+768)] = ((f16(acc3))+val35);
  data0[(alu23+769)] = ((f16(acc7))+val35);
  data0[(alu23+770)] = ((f16(acc11))+val35);
  data0[(alu23+771)] = ((f16(acc15))+val35);
}`;

const r_2_256_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val0 = data1[(gidx0+(gidx1*327680)+(lidx0*20480)+(ridx0<<8))];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[(gidx0+(gidx1<<8))] = (f16((acc1*0.0007812500116415322f)));
  }
}`;

const r_2_256_16_80n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0+(gidx1<<8));
  var val0 = data2[alu0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data1[(gidx0+(gidx1*327680)+(lidx0*20480)+(ridx0<<8))];
    var alu1 = (val1-val0);
    acc0 = (acc0+(f32((alu1*alu1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[alu0] = sqrt((1/((f16((acc1*0.0007812500116415322f)))+(f16(1e-05f)))));
  }
}`;

const E_2_8_20_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*327680);
  var alu1 = (gidx0<<6);
  var alu2 = (gidx1<<5);
  var alu3 = (lidx0<<2);
  var alu4 = (alu2+(gidx2<<8)+alu3);
  var val0 = data2[alu4];
  var val1 = data3[alu4];
  var alu5 = (alu4+1);
  var val2 = data2[alu5];
  var val3 = data3[alu5];
  var alu6 = (alu4+2);
  var val4 = data2[alu6];
  var val5 = data3[alu6];
  var alu7 = (alu4+3);
  var val6 = data2[alu7];
  var val7 = data3[alu7];
  var alu8 = (lidx1<<2);
  var alu9 = (alu1+alu8);
  var val8 = data4[alu9];
  var val9 = data5[alu9];
  var alu10 = (alu9+1);
  var val10 = data4[alu10];
  var val11 = data5[alu10];
  var alu11 = (alu9+2);
  var val12 = data4[alu11];
  var val13 = data5[alu11];
  var alu12 = (alu9+3);
  var val14 = data4[alu12];
  var val15 = data5[alu12];
  var alu13 = (alu2+alu0+(gidx0<<14)+alu3+(lidx1<<10));
  var val16 = data1[alu13];
  var val17 = data1[(alu13+1)];
  var val18 = data1[(alu13+2)];
  var val19 = data1[(alu13+3)];
  var val20 = data1[(alu13+256)];
  var val21 = data1[(alu13+257)];
  var val22 = data1[(alu13+258)];
  var val23 = data1[(alu13+259)];
  var val24 = data1[(alu13+512)];
  var val25 = data1[(alu13+513)];
  var val26 = data1[(alu13+514)];
  var val27 = data1[(alu13+515)];
  var val28 = data1[(alu13+768)];
  var val29 = data1[(alu13+769)];
  var val30 = data1[(alu13+770)];
  var val31 = data1[(alu13+771)];
  var alu14 = ((gidx1*40960)+alu0+alu1+(lidx0*5120)+alu8);
  data0[(alu14+1280)] = (val9+(val8*val3*(val17-val2)));
  data0[(alu14+2560)] = (val9+(val8*val5*(val18-val4)));
  data0[(alu14+3840)] = (val9+(val8*val7*(val19-val6)));
  data0[(alu14+1)] = (val11+(val10*val1*(val20-val0)));
  data0[(alu14+1281)] = (val11+(val10*val3*(val21-val2)));
  data0[(alu14+2561)] = (val11+(val10*val5*(val22-val4)));
  data0[(alu14+3841)] = (val11+(val10*val7*(val23-val6)));
  data0[(alu14+2)] = (val13+(val12*val1*(val24-val0)));
  data0[(alu14+1282)] = (val13+(val12*val3*(val25-val2)));
  data0[(alu14+2562)] = (val13+(val12*val5*(val26-val4)));
  data0[(alu14+3842)] = (val13+(val12*val7*(val27-val6)));
  data0[(alu14+3)] = (val15+(val14*val1*(val28-val0)));
  data0[(alu14+1283)] = (val15+(val14*val3*(val29-val2)));
  data0[(alu14+2563)] = (val15+(val14*val5*(val30-val4)));
  data0[(alu14+3843)] = (val15+(val14*val7*(val31-val6)));
  data0[alu14] = (val9+(val8*val1*(val16-val0)));
}`;

const r_16_20_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1*40960);
  var alu1 = (lidx0*5120);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu2 = (ridx0<<2);
    var alu3 = ((gidx0*81920)+(lidx1*5120)+alu2);
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+1280)];
    var val5 = data2[(alu3+1281)];
    var val6 = data2[(alu3+1282)];
    var val7 = data2[(alu3+1283)];
    var val8 = data2[(alu3+2560)];
    var val9 = data2[(alu3+2561)];
    var val10 = data2[(alu3+2562)];
    var val11 = data2[(alu3+2563)];
    var val12 = data2[(alu3+3840)];
    var val13 = data2[(alu3+3841)];
    var val14 = data2[(alu3+3842)];
    var val15 = data2[(alu3+3843)];
    var alu4 = (alu0+alu1+alu2);
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+1280)];
    var val21 = data1[(alu4+1281)];
    var val22 = data1[(alu4+1282)];
    var val23 = data1[(alu4+1283)];
    var val24 = data1[(alu4+2560)];
    var val25 = data1[(alu4+2561)];
    var val26 = data1[(alu4+2562)];
    var val27 = data1[(alu4+2563)];
    var val28 = data1[(alu4+3840)];
    var val29 = data1[(alu4+3841)];
    var val30 = data1[(alu4+3842)];
    var val31 = data1[(alu4+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu22 = ((gidx0<<6)+alu0+alu1+(lidx1<<2));
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+1280)] = (f16(acc1));
  data0[(alu22+1281)] = (f16(acc5));
  data0[(alu22+1282)] = (f16(acc9));
  data0[(alu22+1283)] = (f16(acc13));
  data0[(alu22+2560)] = (f16(acc2));
  data0[(alu22+2561)] = (f16(acc6));
  data0[(alu22+2562)] = (f16(acc10));
  data0[(alu22+2563)] = (f16(acc14));
  data0[(alu22+3840)] = (f16(acc3));
  data0[(alu22+3841)] = (f16(acc7));
  data0[(alu22+3842)] = (f16(acc11));
  data0[(alu22+3843)] = (f16(acc15));
}`;

const r_2_8_8_4_8_16_40_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0>>2);
  var alu1 = (gidx0&3);
  var alu2 = ((gidx1*160)+(gidx2*327680));
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
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var alu3 = (ridx0<<2);
    var alu4 = (alu2+(alu0*40960)+(lidx0*5120)+alu3);
    var val0 = data1[alu4];
    var val1 = data1[(alu4+1)];
    var val2 = data1[(alu4+2)];
    var val3 = data1[(alu4+3)];
    var val4 = data1[(alu4+1280)];
    var val5 = data1[(alu4+1281)];
    var val6 = data1[(alu4+1282)];
    var val7 = data1[(alu4+1283)];
    var val8 = data1[(alu4+2560)];
    var val9 = data1[(alu4+2561)];
    var val10 = data1[(alu4+2562)];
    var val11 = data1[(alu4+2563)];
    var val12 = data1[(alu4+3840)];
    var val13 = data1[(alu4+3841)];
    var val14 = data1[(alu4+3842)];
    var val15 = data1[(alu4+3843)];
    var alu5 = (alu2+(alu1*81920)+(lidx1*5120)+alu3);
    var val16 = data2[alu5];
    var val17 = data2[(alu5+1)];
    var val18 = data2[(alu5+2)];
    var val19 = data2[(alu5+3)];
    var val20 = data2[(alu5+1280)];
    var val21 = data2[(alu5+1281)];
    var val22 = data2[(alu5+1282)];
    var val23 = data2[(alu5+1283)];
    var val24 = data2[(alu5+2560)];
    var val25 = data2[(alu5+2561)];
    var val26 = data2[(alu5+2562)];
    var val27 = data2[(alu5+2563)];
    var val28 = data2[(alu5+3840)];
    var val29 = data2[(alu5+3841)];
    var val30 = data2[(alu5+3842)];
    var val31 = data2[(alu5+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc2 = (acc2+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc3 = (acc3+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc4 = (acc4+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc7 = (acc7+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc8 = (acc8+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc9 = (acc9+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc12 = (acc12+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc13 = (acc13+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc14 = (acc14+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu23 = ((gidx1<<16)+(gidx2<<19)+(alu0<<13)+(alu1<<6)+(lidx0<<10)+(lidx1<<2));
  data0[alu23] = (acc0*0.07905694097280502f);
  data0[(alu23+1)] = (acc4*0.07905694097280502f);
  data0[(alu23+2)] = (acc8*0.07905694097280502f);
  data0[(alu23+3)] = (acc12*0.07905694097280502f);
  data0[(alu23+256)] = (acc1*0.07905694097280502f);
  data0[(alu23+257)] = (acc5*0.07905694097280502f);
  data0[(alu23+258)] = (acc9*0.07905694097280502f);
  data0[(alu23+259)] = (acc13*0.07905694097280502f);
  data0[(alu23+512)] = (acc2*0.07905694097280502f);
  data0[(alu23+513)] = (acc6*0.07905694097280502f);
  data0[(alu23+514)] = (acc10*0.07905694097280502f);
  data0[(alu23+515)] = (acc14*0.07905694097280502f);
  data0[(alu23+768)] = (acc3*0.07905694097280502f);
  data0[(alu23+769)] = (acc7*0.07905694097280502f);
  data0[(alu23+770)] = (acc11*0.07905694097280502f);
  data0[(alu23+771)] = (acc15*0.07905694097280502f);
}`;

const r_128_32_64_4 = `enable f16;
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
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu0 = ((gidx0<<13)+(lidx0<<8)+(ridx0<<2));
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

const r_32_32_64_4_4 = `enable f16;
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
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu4 = ((gidx0<<15)+(lidx0<<10)+(ridx0<<2));
    var val4 = data1[alu4];
    var val5 = data1[(alu4+1)];
    var val6 = data1[(alu4+2)];
    var val7 = data1[(alu4+3)];
    var val8 = data1[(alu4+256)];
    var val9 = data1[(alu4+257)];
    var val10 = data1[(alu4+258)];
    var val11 = data1[(alu4+259)];
    var val12 = data1[(alu4+512)];
    var val13 = data1[(alu4+513)];
    var val14 = data1[(alu4+514)];
    var val15 = data1[(alu4+515)];
    var val16 = data1[(alu4+768)];
    var val17 = data1[(alu4+769)];
    var val18 = data1[(alu4+770)];
    var val19 = data1[(alu4+771)];
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

const E_512_4_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 512 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (1/val1);
  data0[alu2] = (f16((exp2(((val3-val0)*1.4426950408889634f))*alu5)));
  data0[alu3] = (f16((exp2(((val4-val0)*1.4426950408889634f))*alu5)));
  data0[alu4] = (f16((exp2(((val5-val0)*1.4426950408889634f))*alu5)));
  data0[alu1] = (f16((exp2(((val2-val0)*1.4426950408889634f))*alu5)));
}`;

const r_2_8_4_5_16_8_64_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx2*327680);
  var alu1 = (gidx0/5);
  var alu2 = ((gidx0%5)<<5);
  var alu3 = (lidx1<<2);
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
    var alu4 = ((gidx1*160)+alu0+alu2+alu3+(ridx0*5120));
    var val0 = data2[alu4];
    var val1 = data2[(alu4+1)];
    var val2 = data2[(alu4+2)];
    var val3 = data2[(alu4+3)];
    var val4 = data2[(alu4+1280)];
    var val5 = data2[(alu4+1281)];
    var val6 = data2[(alu4+1282)];
    var val7 = data2[(alu4+1283)];
    var val8 = data2[(alu4+2560)];
    var val9 = data2[(alu4+2561)];
    var val10 = data2[(alu4+2562)];
    var val11 = data2[(alu4+2563)];
    var val12 = data2[(alu4+3840)];
    var val13 = data2[(alu4+3841)];
    var val14 = data2[(alu4+3842)];
    var val15 = data2[(alu4+3843)];
    var alu5 = ((gidx1<<16)+(gidx2<<19)+(alu1<<14)+(lidx0<<10)+(ridx0<<2));
    var val16 = data1[alu5];
    var val17 = data1[(alu5+1)];
    var val18 = data1[(alu5+2)];
    var val19 = data1[(alu5+3)];
    var val20 = data1[(alu5+256)];
    var val21 = data1[(alu5+257)];
    var val22 = data1[(alu5+258)];
    var val23 = data1[(alu5+259)];
    var val24 = data1[(alu5+512)];
    var val25 = data1[(alu5+513)];
    var val26 = data1[(alu5+514)];
    var val27 = data1[(alu5+515)];
    var val28 = data1[(alu5+768)];
    var val29 = data1[(alu5+769)];
    var val30 = data1[(alu5+770)];
    var val31 = data1[(alu5+771)];
    acc0 = (acc0+(f32((val12*val19)))+(f32((val8*val18)))+(f32((val4*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val12*val23)))+(f32((val8*val22)))+(f32((val4*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val12*val27)))+(f32((val8*val26)))+(f32((val4*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val12*val31)))+(f32((val8*val30)))+(f32((val4*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val13*val19)))+(f32((val9*val18)))+(f32((val1*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val13*val23)))+(f32((val9*val22)))+(f32((val1*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val13*val27)))+(f32((val9*val26)))+(f32((val1*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val13*val31)))+(f32((val9*val30)))+(f32((val1*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val14*val19)))+(f32((val10*val18)))+(f32((val2*val16)))+(f32((val6*val17))));
    acc9 = (acc9+(f32((val14*val23)))+(f32((val10*val22)))+(f32((val2*val20)))+(f32((val6*val21))));
    acc10 = (acc10+(f32((val14*val27)))+(f32((val10*val26)))+(f32((val2*val24)))+(f32((val6*val25))));
    acc11 = (acc11+(f32((val14*val31)))+(f32((val10*val30)))+(f32((val2*val28)))+(f32((val6*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val11*val18)))+(f32((val3*val16)))+(f32((val7*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val11*val22)))+(f32((val3*val20)))+(f32((val7*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val11*val26)))+(f32((val3*val24)))+(f32((val7*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val11*val30)))+(f32((val3*val28)))+(f32((val7*val29))));
  }
  var alu23 = ((gidx1*40960)+alu0+(alu1*10240)+alu2+(lidx0*640)+alu3);
  data0[alu23] = (f16(acc0));
  data0[(alu23+1)] = (f16(acc4));
  data0[(alu23+2)] = (f16(acc8));
  data0[(alu23+3)] = (f16(acc12));
  data0[(alu23+160)] = (f16(acc1));
  data0[(alu23+161)] = (f16(acc5));
  data0[(alu23+162)] = (f16(acc9));
  data0[(alu23+163)] = (f16(acc13));
  data0[(alu23+320)] = (f16(acc2));
  data0[(alu23+321)] = (f16(acc6));
  data0[(alu23+322)] = (f16(acc10));
  data0[(alu23+323)] = (f16(acc14));
  data0[(alu23+480)] = (f16(acc3));
  data0[(alu23+481)] = (f16(acc7));
  data0[(alu23+482)] = (f16(acc11));
  data0[(alu23+483)] = (f16(acc15));
}`;

const r_2_8_20_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*327680);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu1 = ((gidx1*5120)+(lidx0*640)+alu0+((ridx0/40)*40960)+((ridx0%40)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+160)];
    var val5 = data2[(alu1+161)];
    var val6 = data2[(alu1+162)];
    var val7 = data2[(alu1+163)];
    var val8 = data2[(alu1+320)];
    var val9 = data2[(alu1+321)];
    var val10 = data2[(alu1+322)];
    var val11 = data2[(alu1+323)];
    var val12 = data2[(alu1+480)];
    var val13 = data2[(alu1+481)];
    var val14 = data2[(alu1+482)];
    var val15 = data2[(alu1+483)];
    var alu2 = ((gidx0*81920)+(lidx1*5120)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+1280)];
    var val21 = data3[(alu2+1281)];
    var val22 = data3[(alu2+1282)];
    var val23 = data3[(alu2+1283)];
    var val24 = data3[(alu2+2560)];
    var val25 = data3[(alu2+2561)];
    var val26 = data3[(alu2+2562)];
    var val27 = data3[(alu2+2563)];
    var val28 = data3[(alu2+3840)];
    var val29 = data3[(alu2+3841)];
    var val30 = data3[(alu2+3842)];
    var val31 = data3[(alu2+3843)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data4[alu22];
  var val33 = data4[(alu22+1)];
  var val34 = data4[(alu22+2)];
  var val35 = data4[(alu22+3)];
  var alu23 = ((gidx1<<5)+alu0+(gidx0<<14)+(lidx0<<2)+(lidx1<<10));
  var val36 = data1[alu23];
  var val37 = data1[(alu23+1)];
  var val38 = data1[(alu23+2)];
  var val39 = data1[(alu23+3)];
  var val40 = data1[(alu23+256)];
  var val41 = data1[(alu23+257)];
  var val42 = data1[(alu23+258)];
  var val43 = data1[(alu23+259)];
  var val44 = data1[(alu23+512)];
  var val45 = data1[(alu23+513)];
  var val46 = data1[(alu23+514)];
  var val47 = data1[(alu23+515)];
  var val48 = data1[(alu23+768)];
  var val49 = data1[(alu23+769)];
  var val50 = data1[(alu23+770)];
  var val51 = data1[(alu23+771)];
  var alu24 = ((gidx1*40960)+alu0+alu20+(lidx0*5120)+alu21);
  data0[alu24] = (val36+(f16(acc0))+val32);
  data0[(alu24+1)] = (val40+(f16(acc4))+val33);
  data0[(alu24+2)] = (val44+(f16(acc8))+val34);
  data0[(alu24+3)] = (val48+(f16(acc12))+val35);
  data0[(alu24+1280)] = (val37+(f16(acc1))+val32);
  data0[(alu24+1281)] = (val41+(f16(acc5))+val33);
  data0[(alu24+1282)] = (val45+(f16(acc9))+val34);
  data0[(alu24+1283)] = (val49+(f16(acc13))+val35);
  data0[(alu24+2560)] = (val38+(f16(acc2))+val32);
  data0[(alu24+2561)] = (val42+(f16(acc6))+val33);
  data0[(alu24+2562)] = (val46+(f16(acc10))+val34);
  data0[(alu24+2563)] = (val50+(f16(acc14))+val35);
  data0[(alu24+3840)] = (val39+(f16(acc3))+val32);
  data0[(alu24+3841)] = (val43+(f16(acc7))+val33);
  data0[(alu24+3842)] = (val47+(f16(acc11))+val34);
  data0[(alu24+3843)] = (val51+(f16(acc15))+val35);
}`;

const r_512_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val0 = data1[((gidx0*1280)+(lidx0*80)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*0.0007812500116415322f)));
  }
}`;

const r_512_16_80n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data1[((gidx0*1280)+(lidx0*80)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*0.0007812500116415322f)))+(f16(1e-05f)))));
  }
}`;

const E_16_20_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data2[alu1];
  var val1 = data3[alu1];
  var alu2 = (alu1+1);
  var val2 = data2[alu2];
  var val3 = data3[alu2];
  var alu3 = (alu1+2);
  var val4 = data2[alu3];
  var val5 = data3[alu3];
  var alu4 = (alu1+3);
  var val6 = data2[alu4];
  var val7 = data3[alu4];
  var alu5 = (lidx1<<2);
  var alu6 = (alu0+(gidx1*40960)+(lidx0*5120)+alu5);
  var val8 = data1[alu6];
  var alu7 = (alu6+1);
  var val9 = data1[alu7];
  var alu8 = (alu6+2);
  var val10 = data1[alu8];
  var alu9 = (alu6+3);
  var val11 = data1[alu9];
  var alu10 = (alu6+1280);
  var val12 = data1[alu10];
  var alu11 = (alu6+1281);
  var val13 = data1[alu11];
  var alu12 = (alu6+1282);
  var val14 = data1[alu12];
  var alu13 = (alu6+1283);
  var val15 = data1[alu13];
  var alu14 = (alu6+2560);
  var val16 = data1[alu14];
  var alu15 = (alu6+2561);
  var val17 = data1[alu15];
  var alu16 = (alu6+2562);
  var val18 = data1[alu16];
  var alu17 = (alu6+2563);
  var val19 = data1[alu17];
  var alu18 = (alu6+3840);
  var val20 = data1[alu18];
  var alu19 = (alu6+3841);
  var val21 = data1[alu19];
  var alu20 = (alu6+3842);
  var val22 = data1[alu20];
  var alu21 = (alu6+3843);
  var val23 = data1[alu21];
  var alu22 = (alu0+alu5);
  var val24 = data4[alu22];
  var val25 = data5[alu22];
  var alu23 = (alu22+1);
  var val26 = data4[alu23];
  var val27 = data5[alu23];
  var alu24 = (alu22+2);
  var val28 = data4[alu24];
  var val29 = data5[alu24];
  var alu25 = (alu22+3);
  var val30 = data4[alu25];
  var val31 = data5[alu25];
  data0[alu7] = (val27+(val26*val1*(val9-val0)));
  data0[alu8] = (val29+(val28*val1*(val10-val0)));
  data0[alu9] = (val31+(val30*val1*(val11-val0)));
  data0[alu10] = (val25+(val24*val3*(val12-val2)));
  data0[alu11] = (val27+(val26*val3*(val13-val2)));
  data0[alu12] = (val29+(val28*val3*(val14-val2)));
  data0[alu13] = (val31+(val30*val3*(val15-val2)));
  data0[alu14] = (val25+(val24*val5*(val16-val4)));
  data0[alu15] = (val27+(val26*val5*(val17-val4)));
  data0[alu16] = (val29+(val28*val5*(val18-val4)));
  data0[alu17] = (val31+(val30*val5*(val19-val4)));
  data0[alu18] = (val25+(val24*val7*(val20-val6)));
  data0[alu19] = (val27+(val26*val7*(val21-val6)));
  data0[alu20] = (val29+(val28*val7*(val22-val6)));
  data0[alu21] = (val31+(val30*val7*(val23-val6)));
  data0[alu6] = (val25+(val24*val1*(val8-val0)));
}`;

const r_2_4_77_8_16_40_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 4 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0*160);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx1*81920)+(gidx2*327680)+alu0+(lidx1*5120)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+1280)];
    var val5 = data1[(alu2+1281)];
    var val6 = data1[(alu2+1282)];
    var val7 = data1[(alu2+1283)];
    var val8 = data1[(alu2+2560)];
    var val9 = data1[(alu2+2561)];
    var val10 = data1[(alu2+2562)];
    var val11 = data1[(alu2+2563)];
    var val12 = data1[(alu2+3840)];
    var val13 = data1[(alu2+3841)];
    var val14 = data1[(alu2+3842)];
    var val15 = data1[(alu2+3843)];
    var alu3 = ((gidx0*1280)+(gidx2*98560)+alu0+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
  }
  var alu9 = (gidx0+(gidx1*4928)+(gidx2*157696)+(lidx0*19712)+(lidx1*308));
  data0[alu9] = (acc0*0.07905694097280502f);
  data0[(alu9+77)] = (acc1*0.07905694097280502f);
  data0[(alu9+154)] = (acc2*0.07905694097280502f);
  data0[(alu9+231)] = (acc3*0.07905694097280502f);
}`;

const r_128_32_77 = `enable f16;
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val0 = data1[((gidx0*2464)+(lidx0*77)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_32_32_77_4 = `enable f16;
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu4 = ((gidx0*9856)+(lidx0*308)+ridx0);
    var val4 = data1[alu4];
    var val5 = data1[(alu4+77)];
    var val6 = data1[(alu4+154)];
    var val7 = data1[(alu4+231)];
    acc0 = (acc0+exp2(((val4-val3)*1.4426950408889634f)));
    acc1 = (acc1+exp2(((val5-val0)*1.4426950408889634f)));
    acc2 = (acc2+exp2(((val6-val1)*1.4426950408889634f)));
    acc3 = (acc3+exp2(((val7-val2)*1.4426950408889634f)));
  }
  data0[alu1] = acc1;
  data0[alu2] = acc2;
  data0[alu3] = acc3;
  data0[alu0] = acc0;
}`;

const E_32_77_32_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (gidx0+(gidx1*9856)+(lidx0*308));
  var val0 = data1[alu0];
  var alu1 = (alu0+77);
  var val1 = data1[alu1];
  var alu2 = (alu0+154);
  var val2 = data1[alu2];
  var alu3 = (alu0+231);
  var val3 = data1[alu3];
  var alu4 = ((gidx1<<7)+(lidx0<<2));
  var val4 = data2[alu4];
  var val5 = data3[alu4];
  var alu5 = (alu4+1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = (alu4+2);
  var val8 = data2[alu6];
  var val9 = data3[alu6];
  var alu7 = (alu4+3);
  var val10 = data2[alu7];
  var val11 = data3[alu7];
  data0[alu0] = (f16((exp2(((val0-val4)*1.4426950408889634f))*(1/val5))));
  data0[alu1] = (f16((exp2(((val1-val6)*1.4426950408889634f))*(1/val7))));
  data0[alu2] = (f16((exp2(((val2-val8)*1.4426950408889634f))*(1/val9))));
  data0[alu3] = (f16((exp2(((val3-val10)*1.4426950408889634f))*(1/val11))));
}`;

const r_2_8_4_5_16_8_77_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx0/5);
  var alu1 = ((gidx0%5)<<5);
  var alu2 = (lidx1<<2);
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu3 = ((gidx1*19712)+(gidx2*157696)+(alu0*4928)+(lidx0*308)+ridx0);
    var val0 = data1[alu3];
    var val1 = data1[(alu3+77)];
    var val2 = data1[(alu3+154)];
    var val3 = data1[(alu3+231)];
    var alu4 = ((gidx1*160)+(gidx2*98560)+alu1+alu2+(ridx0*1280));
    var val4 = data2[alu4];
    var val5 = data2[(alu4+1)];
    var val6 = data2[(alu4+2)];
    var val7 = data2[(alu4+3)];
    acc0 = (acc0+(f32((val4*val0))));
    acc1 = (acc1+(f32((val4*val1))));
    acc2 = (acc2+(f32((val4*val2))));
    acc3 = (acc3+(f32((val4*val3))));
    acc4 = (acc4+(f32((val5*val0))));
    acc5 = (acc5+(f32((val5*val1))));
    acc6 = (acc6+(f32((val5*val2))));
    acc7 = (acc7+(f32((val5*val3))));
    acc8 = (acc8+(f32((val6*val0))));
    acc9 = (acc9+(f32((val6*val1))));
    acc10 = (acc10+(f32((val6*val2))));
    acc11 = (acc11+(f32((val6*val3))));
    acc12 = (acc12+(f32((val7*val0))));
    acc13 = (acc13+(f32((val7*val1))));
    acc14 = (acc14+(f32((val7*val2))));
    acc15 = (acc15+(f32((val7*val3))));
  }
  var alu22 = ((gidx1*40960)+(gidx2*327680)+(alu0*10240)+alu1+(lidx0*640)+alu2);
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+160)] = (f16(acc1));
  data0[(alu22+161)] = (f16(acc5));
  data0[(alu22+162)] = (f16(acc9));
  data0[(alu22+163)] = (f16(acc13));
  data0[(alu22+320)] = (f16(acc2));
  data0[(alu22+321)] = (f16(acc6));
  data0[(alu22+322)] = (f16(acc10));
  data0[(alu22+323)] = (f16(acc14));
  data0[(alu22+480)] = (f16(acc3));
  data0[(alu22+481)] = (f16(acc7));
  data0[(alu22+482)] = (f16(acc11));
  data0[(alu22+483)] = (f16(acc15));
}`;

const r_2_8_20_8_16_320_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*327680);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu1 = ((gidx1*5120)+(lidx0*640)+alu0+((ridx0/40)*40960)+((ridx0%40)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+160)];
    var val5 = data2[(alu1+161)];
    var val6 = data2[(alu1+162)];
    var val7 = data2[(alu1+163)];
    var val8 = data2[(alu1+320)];
    var val9 = data2[(alu1+321)];
    var val10 = data2[(alu1+322)];
    var val11 = data2[(alu1+323)];
    var val12 = data2[(alu1+480)];
    var val13 = data2[(alu1+481)];
    var val14 = data2[(alu1+482)];
    var val15 = data2[(alu1+483)];
    var alu2 = ((gidx0*81920)+(lidx1*5120)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+1280)];
    var val21 = data3[(alu2+1281)];
    var val22 = data3[(alu2+1282)];
    var val23 = data3[(alu2+1283)];
    var val24 = data3[(alu2+2560)];
    var val25 = data3[(alu2+2561)];
    var val26 = data3[(alu2+2562)];
    var val27 = data3[(alu2+2563)];
    var val28 = data3[(alu2+3840)];
    var val29 = data3[(alu2+3841)];
    var val30 = data3[(alu2+3842)];
    var val31 = data3[(alu2+3843)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = ((gidx1*40960)+alu0+alu20+(lidx0*5120)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+1280);
  var val36 = data1[alu26];
  var alu27 = (alu22+1281);
  var val37 = data1[alu27];
  var alu28 = (alu22+1282);
  var val38 = data1[alu28];
  var alu29 = (alu22+1283);
  var val39 = data1[alu29];
  var alu30 = (alu22+2560);
  var val40 = data1[alu30];
  var alu31 = (alu22+2561);
  var val41 = data1[alu31];
  var alu32 = (alu22+2562);
  var val42 = data1[alu32];
  var alu33 = (alu22+2563);
  var val43 = data1[alu33];
  var alu34 = (alu22+3840);
  var val44 = data1[alu34];
  var alu35 = (alu22+3841);
  var val45 = data1[alu35];
  var alu36 = (alu22+3842);
  var val46 = data1[alu36];
  var alu37 = (alu22+3843);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_16_160_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*81920)+(lidx1*5120)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+1280)];
    var val5 = data2[(alu1+1281)];
    var val6 = data2[(alu1+1282)];
    var val7 = data2[(alu1+1283)];
    var val8 = data2[(alu1+2560)];
    var val9 = data2[(alu1+2561)];
    var val10 = data2[(alu1+2562)];
    var val11 = data2[(alu1+2563)];
    var val12 = data2[(alu1+3840)];
    var val13 = data2[(alu1+3841)];
    var val14 = data2[(alu1+3842)];
    var val15 = data2[(alu1+3843)];
    var alu2 = ((gidx1*40960)+(lidx0*5120)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+1280)];
    var val21 = data1[(alu2+1281)];
    var val22 = data1[(alu2+1282)];
    var val23 = data1[(alu2+1283)];
    var val24 = data1[(alu2+2560)];
    var val25 = data1[(alu2+2561)];
    var val26 = data1[(alu2+2562)];
    var val27 = data1[(alu2+2563)];
    var val28 = data1[(alu2+3840)];
    var val29 = data1[(alu2+3841)];
    var val30 = data1[(alu2+3842)];
    var val31 = data1[(alu2+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = (alu20+(gidx1*327680)+(lidx0*40960)+alu21);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val33);
  data0[(alu23+2)] = ((f16(acc8))+val34);
  data0[(alu23+3)] = ((f16(acc12))+val35);
  data0[(alu23+10240)] = ((f16(acc1))+val32);
  data0[(alu23+10241)] = ((f16(acc5))+val33);
  data0[(alu23+10242)] = ((f16(acc9))+val34);
  data0[(alu23+10243)] = ((f16(acc13))+val35);
  data0[(alu23+20480)] = ((f16(acc2))+val32);
  data0[(alu23+20481)] = ((f16(acc6))+val33);
  data0[(alu23+20482)] = ((f16(acc10))+val34);
  data0[(alu23+20483)] = ((f16(acc14))+val35);
  data0[(alu23+30720)] = ((f16(acc3))+val32);
  data0[(alu23+30721)] = ((f16(acc7))+val33);
  data0[(alu23+30722)] = ((f16(acc11))+val34);
  data0[(alu23+30723)] = ((f16(acc15))+val35);
}`;

const E_64_80_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 80 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = (alu0+(gidx1*81920)+(lidx0*10240)+alu1);
  var val0 = data1[alu2];
  var val1 = data1[(alu2+1)];
  var val2 = data1[(alu2+2)];
  var val3 = data1[(alu2+3)];
  var val4 = data1[(alu2+5120)];
  var val5 = data1[(alu2+5121)];
  var val6 = data1[(alu2+5122)];
  var val7 = data1[(alu2+5123)];
  var alu3 = (alu0+(gidx1*40960)+(lidx0*5120)+alu1);
  data0[alu3] = (val0*(1/(exp2(((val4+(val4*val4*val4*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val4);
  data0[(alu3+1)] = (val1*(1/(exp2(((val5+(val5*val5*val5*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val5);
  data0[(alu3+2)] = (val2*(1/(exp2(((val6+(val6*val6*val6*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val6);
  data0[(alu3+3)] = (val3*(1/(exp2(((val7+(val7*val7*val7*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val7);
}`;

const r_16_20_8_16_1280_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*327680)+(lidx1*20480)+alu0);
    var val0 = data3[alu1];
    var val1 = data3[(alu1+1)];
    var val2 = data3[(alu1+2)];
    var val3 = data3[(alu1+3)];
    var val4 = data3[(alu1+5120)];
    var val5 = data3[(alu1+5121)];
    var val6 = data3[(alu1+5122)];
    var val7 = data3[(alu1+5123)];
    var val8 = data3[(alu1+10240)];
    var val9 = data3[(alu1+10241)];
    var val10 = data3[(alu1+10242)];
    var val11 = data3[(alu1+10243)];
    var val12 = data3[(alu1+15360)];
    var val13 = data3[(alu1+15361)];
    var val14 = data3[(alu1+15362)];
    var val15 = data3[(alu1+15363)];
    var alu2 = ((gidx1*163840)+(lidx0*20480)+alu0);
    var val16 = data2[alu2];
    var val17 = data2[(alu2+1)];
    var val18 = data2[(alu2+2)];
    var val19 = data2[(alu2+3)];
    var val20 = data2[(alu2+5120)];
    var val21 = data2[(alu2+5121)];
    var val22 = data2[(alu2+5122)];
    var val23 = data2[(alu2+5123)];
    var val24 = data2[(alu2+10240)];
    var val25 = data2[(alu2+10241)];
    var val26 = data2[(alu2+10242)];
    var val27 = data2[(alu2+10243)];
    var val28 = data2[(alu2+15360)];
    var val29 = data2[(alu2+15361)];
    var val30 = data2[(alu2+15362)];
    var val31 = data2[(alu2+15363)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+(gidx1*40960)+(lidx0*5120)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+1280);
  var val36 = data1[alu26];
  var alu27 = (alu22+1281);
  var val37 = data1[alu27];
  var alu28 = (alu22+1282);
  var val38 = data1[alu28];
  var alu29 = (alu22+1283);
  var val39 = data1[alu29];
  var alu30 = (alu22+2560);
  var val40 = data1[alu30];
  var alu31 = (alu22+2561);
  var val41 = data1[alu31];
  var alu32 = (alu22+2562);
  var val42 = data1[alu32];
  var alu33 = (alu22+2563);
  var val43 = data1[alu33];
  var alu34 = (alu22+3840);
  var val44 = data1[alu34];
  var alu35 = (alu22+3841);
  var val45 = data1[alu35];
  var alu36 = (alu22+3842);
  var val46 = data1[alu36];
  var alu37 = (alu22+3843);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_2_40_4_8_16_320_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*327680);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*81920)+alu0+(lidx1*5120)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+1280)];
    var val5 = data1[(alu2+1281)];
    var val6 = data1[(alu2+1282)];
    var val7 = data1[(alu2+1283)];
    var val8 = data1[(alu2+2560)];
    var val9 = data1[(alu2+2561)];
    var val10 = data1[(alu2+2562)];
    var val11 = data1[(alu2+2563)];
    var val12 = data1[(alu2+3840)];
    var val13 = data1[(alu2+3841)];
    var val14 = data1[(alu2+3842)];
    var val15 = data1[(alu2+3843)];
    var alu3 = ((gidx1*40960)+(lidx0*5120)+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    var val20 = data2[(alu3+1280)];
    var val21 = data2[(alu3+1281)];
    var val22 = data2[(alu3+1282)];
    var val23 = data2[(alu3+1283)];
    var val24 = data2[(alu3+2560)];
    var val25 = data2[(alu3+2561)];
    var val26 = data2[(alu3+2562)];
    var val27 = data2[(alu3+2563)];
    var val28 = data2[(alu3+3840)];
    var val29 = data2[(alu3+3841)];
    var val30 = data2[(alu3+3842)];
    var val31 = data2[(alu3+3843)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu21 = ((gidx1<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data3[(alu21+1)];
  var val34 = data3[(alu21+2)];
  var val35 = data3[(alu21+3)];
  var alu22 = ((gidx1<<13)+alu0+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val36 = data4[alu22];
  var alu23 = (alu22+1);
  var val37 = data4[alu23];
  var alu24 = (alu22+2);
  var val38 = data4[alu24];
  var alu25 = (alu22+3);
  var val39 = data4[alu25];
  var alu26 = (alu22+256);
  var val40 = data4[alu26];
  var alu27 = (alu22+257);
  var val41 = data4[alu27];
  var alu28 = (alu22+258);
  var val42 = data4[alu28];
  var alu29 = (alu22+259);
  var val43 = data4[alu29];
  var alu30 = (alu22+512);
  var val44 = data4[alu30];
  var alu31 = (alu22+513);
  var val45 = data4[alu31];
  var alu32 = (alu22+514);
  var val46 = data4[alu32];
  var alu33 = (alu22+515);
  var val47 = data4[alu33];
  var alu34 = (alu22+768);
  var val48 = data4[alu34];
  var alu35 = (alu22+769);
  var val49 = data4[alu35];
  var alu36 = (alu22+770);
  var val50 = data4[alu36];
  var alu37 = (alu22+771);
  var val51 = data4[alu37];
  data0[alu23] = (val37+(f16(acc1))+val32);
  data0[alu24] = (val38+(f16(acc2))+val32);
  data0[alu25] = (val39+(f16(acc3))+val32);
  data0[alu26] = (val40+(f16(acc4))+val33);
  data0[alu27] = (val41+(f16(acc5))+val33);
  data0[alu28] = (val42+(f16(acc6))+val33);
  data0[alu29] = (val43+(f16(acc7))+val33);
  data0[alu30] = (val44+(f16(acc8))+val34);
  data0[alu31] = (val45+(f16(acc9))+val34);
  data0[alu32] = (val46+(f16(acc10))+val34);
  data0[alu33] = (val47+(f16(acc11))+val34);
  data0[alu34] = (val48+(f16(acc12))+val35);
  data0[alu35] = (val49+(f16(acc13))+val35);
  data0[alu36] = (val50+(f16(acc14))+val35);
  data0[alu37] = (val51+(f16(acc15))+val35);
  data0[alu22] = (val36+(f16(acc0))+val32);
}`;

const r_2_160_2_16_4_1280_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (gidx1*327680);
  var alu1 = (lidx1<<4);
  var alu2 = (lidx2<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = ((lidx2<1)!=true);
  var alu5 = (lidx1<15);
  var alu6 = (lidx2<3);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx0*92160)+(lidx0*46080)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+11520)];
    var val10 = data2[(alu7+11521)];
    var val11 = data2[(alu7+11522)];
    var val12 = data2[(alu7+11523)];
    var val13 = data2[(alu7+11524)];
    var val14 = data2[(alu7+11525)];
    var val15 = data2[(alu7+11526)];
    var val16 = data2[(alu7+11527)];
    var val17 = data2[(alu7+11528)];
    var val18 = data2[(alu7+23040)];
    var val19 = data2[(alu7+23041)];
    var val20 = data2[(alu7+23042)];
    var val21 = data2[(alu7+23043)];
    var val22 = data2[(alu7+23044)];
    var val23 = data2[(alu7+23045)];
    var val24 = data2[(alu7+23046)];
    var val25 = data2[(alu7+23047)];
    var val26 = data2[(alu7+23048)];
    var val27 = data2[(alu7+34560)];
    var val28 = data2[(alu7+34561)];
    var val29 = data2[(alu7+34562)];
    var val30 = data2[(alu7+34563)];
    var val31 = data2[(alu7+34564)];
    var val32 = data2[(alu7+34565)];
    var val33 = data2[(alu7+34566)];
    var val34 = data2[(alu7+34567)];
    var val35 = data2[(alu7+34568)];
    var alu8 = (alu0+(ridx0<<8)+alu1+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-17)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-16)], alu3);
    var val39 = select((f16(0.0f)), data1[(alu8+-15)], alu3);
    var val40 = select((f16(0.0f)), data1[(alu8+-14)], alu3);
    var val41 = select((f16(0.0f)), data1[(alu8+-13)], alu3);
    var val42 = select((f16(0.0f)), data1[(alu8+-12)], (alu6&alu3));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu4);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+15)], (alu5&alu4));
    var val49 = select((f16(0.0f)), data1[(alu8+16)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+17)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+18)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+19)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+20)], (alu5&alu6));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = ((gidx0<<3)+(lidx0<<2));
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx0<<11)+alu0+(lidx0<<10)+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+256)] = (val57+(f16(acc1))+val56);
  data0[(alu30+257)] = (val57+(f16(acc5))+val56);
  data0[(alu30+258)] = (val57+(f16(acc9))+val56);
  data0[(alu30+259)] = (val57+(f16(acc13))+val56);
  data0[(alu30+512)] = (val59+(f16(acc2))+val58);
  data0[(alu30+513)] = (val59+(f16(acc6))+val58);
  data0[(alu30+514)] = (val59+(f16(acc10))+val58);
  data0[(alu30+515)] = (val59+(f16(acc14))+val58);
  data0[(alu30+768)] = (val61+(f16(acc3))+val60);
  data0[(alu30+769)] = (val61+(f16(acc7))+val60);
  data0[(alu30+770)] = (val61+(f16(acc11))+val60);
  data0[(alu30+771)] = (val61+(f16(acc15))+val60);
}`;

const r_2_160_2_16_4_1280_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (gidx1*327680);
  var alu1 = (lidx1<<4);
  var alu2 = (lidx2<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = ((lidx2<1)!=true);
  var alu5 = (lidx1<15);
  var alu6 = (lidx2<3);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx0*92160)+(lidx0*46080)+(ridx0*9));
    var val0 = data3[alu7];
    var val1 = data3[(alu7+1)];
    var val2 = data3[(alu7+2)];
    var val3 = data3[(alu7+3)];
    var val4 = data3[(alu7+4)];
    var val5 = data3[(alu7+5)];
    var val6 = data3[(alu7+6)];
    var val7 = data3[(alu7+7)];
    var val8 = data3[(alu7+8)];
    var val9 = data3[(alu7+11520)];
    var val10 = data3[(alu7+11521)];
    var val11 = data3[(alu7+11522)];
    var val12 = data3[(alu7+11523)];
    var val13 = data3[(alu7+11524)];
    var val14 = data3[(alu7+11525)];
    var val15 = data3[(alu7+11526)];
    var val16 = data3[(alu7+11527)];
    var val17 = data3[(alu7+11528)];
    var val18 = data3[(alu7+23040)];
    var val19 = data3[(alu7+23041)];
    var val20 = data3[(alu7+23042)];
    var val21 = data3[(alu7+23043)];
    var val22 = data3[(alu7+23044)];
    var val23 = data3[(alu7+23045)];
    var val24 = data3[(alu7+23046)];
    var val25 = data3[(alu7+23047)];
    var val26 = data3[(alu7+23048)];
    var val27 = data3[(alu7+34560)];
    var val28 = data3[(alu7+34561)];
    var val29 = data3[(alu7+34562)];
    var val30 = data3[(alu7+34563)];
    var val31 = data3[(alu7+34564)];
    var val32 = data3[(alu7+34565)];
    var val33 = data3[(alu7+34566)];
    var val34 = data3[(alu7+34567)];
    var val35 = data3[(alu7+34568)];
    var alu8 = (alu0+(ridx0<<8)+alu1+alu2);
    var val36 = data2[alu8];
    var val37 = select((f16(0.0f)), data2[(alu8+-17)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data2[(alu8+-16)], alu3);
    var val39 = select((f16(0.0f)), data2[(alu8+-15)], alu3);
    var val40 = select((f16(0.0f)), data2[(alu8+-14)], alu3);
    var val41 = select((f16(0.0f)), data2[(alu8+-13)], alu3);
    var val42 = select((f16(0.0f)), data2[(alu8+-12)], (alu6&alu3));
    var val43 = select((f16(0.0f)), data2[(alu8+-1)], alu4);
    var val44 = data2[(alu8+1)];
    var val45 = data2[(alu8+2)];
    var val46 = data2[(alu8+3)];
    var val47 = select((f16(0.0f)), data2[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data2[(alu8+15)], (alu5&alu4));
    var val49 = select((f16(0.0f)), data2[(alu8+16)], alu5);
    var val50 = select((f16(0.0f)), data2[(alu8+17)], alu5);
    var val51 = select((f16(0.0f)), data2[(alu8+18)], alu5);
    var val52 = select((f16(0.0f)), data2[(alu8+19)], alu5);
    var val53 = select((f16(0.0f)), data2[(alu8+20)], (alu5&alu6));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = ((gidx0<<3)+(lidx0<<2));
  var val54 = data4[alu26];
  var val55 = data4[(alu26+1)];
  var val56 = data4[(alu26+2)];
  var val57 = data4[(alu26+3)];
  var alu27 = ((gidx0<<11)+alu0+(lidx0<<10)+alu1+alu2);
  var val58 = data1[alu27];
  var alu28 = (alu27+1);
  var val59 = data1[alu28];
  var alu29 = (alu27+2);
  var val60 = data1[alu29];
  var alu30 = (alu27+3);
  var val61 = data1[alu30];
  var alu31 = (alu27+256);
  var val62 = data1[alu31];
  var alu32 = (alu27+257);
  var val63 = data1[alu32];
  var alu33 = (alu27+258);
  var val64 = data1[alu33];
  var alu34 = (alu27+259);
  var val65 = data1[alu34];
  var alu35 = (alu27+512);
  var val66 = data1[alu35];
  var alu36 = (alu27+513);
  var val67 = data1[alu36];
  var alu37 = (alu27+514);
  var val68 = data1[alu37];
  var alu38 = (alu27+515);
  var val69 = data1[alu38];
  var alu39 = (alu27+768);
  var val70 = data1[alu39];
  var alu40 = (alu27+769);
  var val71 = data1[alu40];
  var alu41 = (alu27+770);
  var val72 = data1[alu41];
  var alu42 = (alu27+771);
  var val73 = data1[alu42];
  data0[alu28] = (val59+(f16(acc4))+val54);
  data0[alu29] = (val60+(f16(acc8))+val54);
  data0[alu30] = (val61+(f16(acc12))+val54);
  data0[alu31] = (val62+(f16(acc1))+val55);
  data0[alu32] = (val63+(f16(acc5))+val55);
  data0[alu33] = (val64+(f16(acc9))+val55);
  data0[alu34] = (val65+(f16(acc13))+val55);
  data0[alu35] = (val66+(f16(acc2))+val56);
  data0[alu36] = (val67+(f16(acc6))+val56);
  data0[alu37] = (val68+(f16(acc10))+val56);
  data0[alu38] = (val69+(f16(acc14))+val56);
  data0[alu39] = (val70+(f16(acc3))+val57);
  data0[alu40] = (val71+(f16(acc7))+val57);
  data0[alu41] = (val72+(f16(acc11))+val57);
  data0[alu42] = (val73+(f16(acc15))+val57);
  data0[alu27] = (val58+(f16(acc0))+val54);
}`;

const r_2_40_8_8_2_1280_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = ((lidx1<1)!=true);
  var alu1 = ((lidx2<1)!=true);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu2 = ((gidx0*368640)+(lidx0*46080)+(ridx0*9));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+4)];
    var val5 = data2[(alu2+5)];
    var val6 = data2[(alu2+6)];
    var val7 = data2[(alu2+7)];
    var val8 = data2[(alu2+8)];
    var val9 = data2[(alu2+11520)];
    var val10 = data2[(alu2+11521)];
    var val11 = data2[(alu2+11522)];
    var val12 = data2[(alu2+11523)];
    var val13 = data2[(alu2+11524)];
    var val14 = data2[(alu2+11525)];
    var val15 = data2[(alu2+11526)];
    var val16 = data2[(alu2+11527)];
    var val17 = data2[(alu2+11528)];
    var val18 = data2[(alu2+23040)];
    var val19 = data2[(alu2+23041)];
    var val20 = data2[(alu2+23042)];
    var val21 = data2[(alu2+23043)];
    var val22 = data2[(alu2+23044)];
    var val23 = data2[(alu2+23045)];
    var val24 = data2[(alu2+23046)];
    var val25 = data2[(alu2+23047)];
    var val26 = data2[(alu2+23048)];
    var val27 = data2[(alu2+34560)];
    var val28 = data2[(alu2+34561)];
    var val29 = data2[(alu2+34562)];
    var val30 = data2[(alu2+34563)];
    var val31 = data2[(alu2+34564)];
    var val32 = data2[(alu2+34565)];
    var val33 = data2[(alu2+34566)];
    var val34 = data2[(alu2+34567)];
    var val35 = data2[(alu2+34568)];
    var alu3 = ((gidx1*327680)+(ridx0<<8)+(lidx1<<5));
    var alu4 = (alu3+(lidx2<<3));
    var val36 = data1[alu4];
    var val37 = select((f16(0.0f)), data1[(alu4+-16)], alu0);
    var val38 = select((f16(0.0f)), data1[(alu4+-15)], alu0);
    var val39 = select((f16(0.0f)), data1[(alu4+-14)], alu0);
    var val40 = select((f16(0.0f)), data1[(alu4+-13)], alu0);
    var val41 = select((f16(0.0f)), data1[(alu4+-12)], alu0);
    var val42 = select((f16(0.0f)), data1[(alu4+-11)], alu0);
    var val43 = select((f16(0.0f)), data1[(alu4+-10)], alu0);
    var val44 = select((f16(0.0f)), data1[(alu4+-9)], alu0);
    var val45 = data1[(alu4+1)];
    var val46 = data1[(alu4+2)];
    var val47 = data1[(alu4+3)];
    var val48 = data1[(alu4+4)];
    var val49 = data1[(alu4+5)];
    var val50 = data1[(alu4+6)];
    var val51 = data1[(alu4+7)];
    var val52 = data1[(alu4+16)];
    var val53 = data1[(alu4+17)];
    var val54 = data1[(alu4+18)];
    var val55 = data1[(alu4+19)];
    var val56 = data1[(alu4+20)];
    var val57 = data1[(alu4+21)];
    var val58 = data1[(alu4+22)];
    var val59 = data1[(alu4+23)];
    var val60 = select((f16(0.0f)), data1[(alu3+-9)], (alu0&alu1));
    var val61 = select((f16(0.0f)), data1[(alu3+7)], alu1);
    var val62 = select((f16(0.0f)), data1[(alu3+23)], alu1);
    acc0 = (acc0+(f32((val53*val8)))+(f32((val45*val5)))+(f32((val38*val2)))+(f32((val52*val7)))+(f32((val36*val4)))+(f32((val37*val1)))+(f32((val62*val6)))+(f32((val60*val0)))+(f32((val61*val3))));
    acc1 = (acc1+(f32((val53*val17)))+(f32((val45*val14)))+(f32((val38*val11)))+(f32((val52*val16)))+(f32((val36*val13)))+(f32((val37*val10)))+(f32((val62*val15)))+(f32((val60*val9)))+(f32((val61*val12))));
    acc2 = (acc2+(f32((val53*val26)))+(f32((val45*val23)))+(f32((val38*val20)))+(f32((val52*val25)))+(f32((val36*val22)))+(f32((val37*val19)))+(f32((val62*val24)))+(f32((val60*val18)))+(f32((val61*val21))));
    acc3 = (acc3+(f32((val53*val35)))+(f32((val45*val32)))+(f32((val38*val29)))+(f32((val52*val34)))+(f32((val36*val31)))+(f32((val37*val28)))+(f32((val62*val33)))+(f32((val60*val27)))+(f32((val61*val30))));
    acc4 = (acc4+(f32((val55*val8)))+(f32((val47*val5)))+(f32((val40*val2)))+(f32((val54*val7)))+(f32((val46*val4)))+(f32((val39*val1)))+(f32((val53*val6)))+(f32((val38*val0)))+(f32((val45*val3))));
    acc5 = (acc5+(f32((val55*val17)))+(f32((val47*val14)))+(f32((val40*val11)))+(f32((val54*val16)))+(f32((val46*val13)))+(f32((val39*val10)))+(f32((val53*val15)))+(f32((val38*val9)))+(f32((val45*val12))));
    acc6 = (acc6+(f32((val55*val26)))+(f32((val47*val23)))+(f32((val40*val20)))+(f32((val54*val25)))+(f32((val46*val22)))+(f32((val39*val19)))+(f32((val53*val24)))+(f32((val38*val18)))+(f32((val45*val21))));
    acc7 = (acc7+(f32((val55*val35)))+(f32((val47*val32)))+(f32((val40*val29)))+(f32((val54*val34)))+(f32((val46*val31)))+(f32((val39*val28)))+(f32((val53*val33)))+(f32((val38*val27)))+(f32((val45*val30))));
    acc8 = (acc8+(f32((val57*val8)))+(f32((val49*val5)))+(f32((val42*val2)))+(f32((val56*val7)))+(f32((val48*val4)))+(f32((val41*val1)))+(f32((val55*val6)))+(f32((val40*val0)))+(f32((val47*val3))));
    acc9 = (acc9+(f32((val57*val17)))+(f32((val49*val14)))+(f32((val42*val11)))+(f32((val56*val16)))+(f32((val48*val13)))+(f32((val41*val10)))+(f32((val55*val15)))+(f32((val40*val9)))+(f32((val47*val12))));
    acc10 = (acc10+(f32((val57*val26)))+(f32((val49*val23)))+(f32((val42*val20)))+(f32((val56*val25)))+(f32((val48*val22)))+(f32((val41*val19)))+(f32((val55*val24)))+(f32((val40*val18)))+(f32((val47*val21))));
    acc11 = (acc11+(f32((val57*val35)))+(f32((val49*val32)))+(f32((val42*val29)))+(f32((val56*val34)))+(f32((val48*val31)))+(f32((val41*val28)))+(f32((val55*val33)))+(f32((val40*val27)))+(f32((val47*val30))));
    acc12 = (acc12+(f32((val59*val8)))+(f32((val51*val5)))+(f32((val44*val2)))+(f32((val58*val7)))+(f32((val50*val4)))+(f32((val43*val1)))+(f32((val57*val6)))+(f32((val42*val0)))+(f32((val49*val3))));
    acc13 = (acc13+(f32((val59*val17)))+(f32((val51*val14)))+(f32((val44*val11)))+(f32((val58*val16)))+(f32((val50*val13)))+(f32((val43*val10)))+(f32((val57*val15)))+(f32((val42*val9)))+(f32((val49*val12))));
    acc14 = (acc14+(f32((val59*val26)))+(f32((val51*val23)))+(f32((val44*val20)))+(f32((val58*val25)))+(f32((val50*val22)))+(f32((val43*val19)))+(f32((val57*val24)))+(f32((val42*val18)))+(f32((val49*val21))));
    acc15 = (acc15+(f32((val59*val35)))+(f32((val51*val32)))+(f32((val44*val29)))+(f32((val58*val34)))+(f32((val50*val31)))+(f32((val43*val28)))+(f32((val57*val33)))+(f32((val42*val27)))+(f32((val49*val30))));
  }
  var alu22 = ((gidx0<<5)+(lidx0<<2));
  var val63 = data3[alu22];
  var val64 = data3[(alu22+1)];
  var val65 = data3[(alu22+2)];
  var val66 = data3[(alu22+3)];
  var alu23 = ((gidx0<<11)+(gidx1*81920)+(lidx0<<8)+(lidx1<<3)+(lidx2<<2));
  data0[alu23] = ((f16(acc0))+val63);
  data0[(alu23+1)] = ((f16(acc4))+val63);
  data0[(alu23+2)] = ((f16(acc8))+val63);
  data0[(alu23+3)] = ((f16(acc12))+val63);
  data0[(alu23+64)] = ((f16(acc1))+val64);
  data0[(alu23+65)] = ((f16(acc5))+val64);
  data0[(alu23+66)] = ((f16(acc9))+val64);
  data0[(alu23+67)] = ((f16(acc13))+val64);
  data0[(alu23+128)] = ((f16(acc2))+val65);
  data0[(alu23+129)] = ((f16(acc6))+val65);
  data0[(alu23+130)] = ((f16(acc10))+val65);
  data0[(alu23+131)] = ((f16(acc14))+val65);
  data0[(alu23+192)] = ((f16(acc3))+val66);
  data0[(alu23+193)] = ((f16(acc7))+val66);
  data0[(alu23+194)] = ((f16(acc11))+val66);
  data0[(alu23+195)] = ((f16(acc15))+val66);
}`;

const r_64_16_160 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var val0 = data1[((gidx0*2560)+(lidx0*160)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*0.0003906250058207661f)));
  }
}`;

const r_64_16_160n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var val1 = data1[((gidx0*2560)+(lidx0*160)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*0.0003906250058207661f)))+(f16(1e-05f)))));
  }
}`;

const E_2_160_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx0<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx1<<5)+(gidx0/5));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx0<<9)+(gidx1*81920)+(lidx0<<6)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_40_8_8_2_1280_4_4_3_3n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (gidx1*81920);
  var alu1 = (lidx1<<3);
  var alu2 = (lidx2<<2);
  var alu3 = (lidx2<1);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (alu3!=true);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx0*368640)+(lidx0*46080)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+11520)];
    var val10 = data2[(alu7+11521)];
    var val11 = data2[(alu7+11522)];
    var val12 = data2[(alu7+11523)];
    var val13 = data2[(alu7+11524)];
    var val14 = data2[(alu7+11525)];
    var val15 = data2[(alu7+11526)];
    var val16 = data2[(alu7+11527)];
    var val17 = data2[(alu7+11528)];
    var val18 = data2[(alu7+23040)];
    var val19 = data2[(alu7+23041)];
    var val20 = data2[(alu7+23042)];
    var val21 = data2[(alu7+23043)];
    var val22 = data2[(alu7+23044)];
    var val23 = data2[(alu7+23045)];
    var val24 = data2[(alu7+23046)];
    var val25 = data2[(alu7+23047)];
    var val26 = data2[(alu7+23048)];
    var val27 = data2[(alu7+34560)];
    var val28 = data2[(alu7+34561)];
    var val29 = data2[(alu7+34562)];
    var val30 = data2[(alu7+34563)];
    var val31 = data2[(alu7+34564)];
    var val32 = data2[(alu7+34565)];
    var val33 = data2[(alu7+34566)];
    var val34 = data2[(alu7+34567)];
    var val35 = data2[(alu7+34568)];
    var alu8 = (alu0+(ridx0<<6)+alu1);
    var alu9 = (alu8+alu2);
    var val36 = data1[alu9];
    var val37 = select((f16(0.0f)), data1[(alu9+-8)], alu4);
    var val38 = select((f16(0.0f)), data1[(alu9+-7)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu9+-6)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu9+-5)], alu4);
    var val41 = data1[(alu9+1)];
    var val42 = data1[(alu9+2)];
    var val43 = data1[(alu9+3)];
    var val44 = select((f16(0.0f)), data1[(alu9+8)], alu6);
    var val45 = select((f16(0.0f)), data1[(alu9+9)], alu6);
    var val46 = select((f16(0.0f)), data1[(alu9+10)], alu6);
    var val47 = select((f16(0.0f)), data1[(alu9+11)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+-5)], (alu4&alu5));
    var val49 = select((f16(0.0f)), data1[(alu8+-4)], (alu3&alu4));
    var val50 = select((f16(0.0f)), data1[(alu8+3)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+4)], alu3);
    var val52 = select((f16(0.0f)), data1[(alu8+11)], (alu6&alu5));
    var val53 = select((f16(0.0f)), data1[(alu8+12)], (alu6&alu3));
    acc0 = (acc0+(f32((val45*val8)))+(f32((val41*val5)))+(f32((val38*val2)))+(f32((val44*val7)))+(f32((val36*val4)))+(f32((val37*val1)))+(f32((val52*val6)))+(f32((val48*val0)))+(f32((val50*val3))));
    acc1 = (acc1+(f32((val45*val17)))+(f32((val41*val14)))+(f32((val38*val11)))+(f32((val44*val16)))+(f32((val36*val13)))+(f32((val37*val10)))+(f32((val52*val15)))+(f32((val48*val9)))+(f32((val50*val12))));
    acc2 = (acc2+(f32((val45*val26)))+(f32((val41*val23)))+(f32((val38*val20)))+(f32((val44*val25)))+(f32((val36*val22)))+(f32((val37*val19)))+(f32((val52*val24)))+(f32((val48*val18)))+(f32((val50*val21))));
    acc3 = (acc3+(f32((val45*val35)))+(f32((val41*val32)))+(f32((val38*val29)))+(f32((val44*val34)))+(f32((val36*val31)))+(f32((val37*val28)))+(f32((val52*val33)))+(f32((val48*val27)))+(f32((val50*val30))));
    acc4 = (acc4+(f32((val46*val8)))+(f32((val42*val5)))+(f32((val39*val2)))+(f32((val45*val7)))+(f32((val41*val4)))+(f32((val38*val1)))+(f32((val44*val6)))+(f32((val37*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val46*val17)))+(f32((val42*val14)))+(f32((val39*val11)))+(f32((val45*val16)))+(f32((val41*val13)))+(f32((val38*val10)))+(f32((val44*val15)))+(f32((val37*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val46*val26)))+(f32((val42*val23)))+(f32((val39*val20)))+(f32((val45*val25)))+(f32((val41*val22)))+(f32((val38*val19)))+(f32((val44*val24)))+(f32((val37*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val46*val35)))+(f32((val42*val32)))+(f32((val39*val29)))+(f32((val45*val34)))+(f32((val41*val31)))+(f32((val38*val28)))+(f32((val44*val33)))+(f32((val37*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val47*val8)))+(f32((val43*val5)))+(f32((val40*val2)))+(f32((val46*val7)))+(f32((val42*val4)))+(f32((val39*val1)))+(f32((val45*val6)))+(f32((val38*val0)))+(f32((val41*val3))));
    acc9 = (acc9+(f32((val47*val17)))+(f32((val43*val14)))+(f32((val40*val11)))+(f32((val46*val16)))+(f32((val42*val13)))+(f32((val39*val10)))+(f32((val45*val15)))+(f32((val38*val9)))+(f32((val41*val12))));
    acc10 = (acc10+(f32((val47*val26)))+(f32((val43*val23)))+(f32((val40*val20)))+(f32((val46*val25)))+(f32((val42*val22)))+(f32((val39*val19)))+(f32((val45*val24)))+(f32((val38*val18)))+(f32((val41*val21))));
    acc11 = (acc11+(f32((val47*val35)))+(f32((val43*val32)))+(f32((val40*val29)))+(f32((val46*val34)))+(f32((val42*val31)))+(f32((val39*val28)))+(f32((val45*val33)))+(f32((val38*val27)))+(f32((val41*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val51*val5)))+(f32((val49*val2)))+(f32((val47*val7)))+(f32((val43*val4)))+(f32((val40*val1)))+(f32((val46*val6)))+(f32((val39*val0)))+(f32((val42*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val51*val14)))+(f32((val49*val11)))+(f32((val47*val16)))+(f32((val43*val13)))+(f32((val40*val10)))+(f32((val46*val15)))+(f32((val39*val9)))+(f32((val42*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val51*val23)))+(f32((val49*val20)))+(f32((val47*val25)))+(f32((val43*val22)))+(f32((val40*val19)))+(f32((val46*val24)))+(f32((val39*val18)))+(f32((val42*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val51*val32)))+(f32((val49*val29)))+(f32((val47*val34)))+(f32((val43*val31)))+(f32((val40*val28)))+(f32((val46*val33)))+(f32((val39*val27)))+(f32((val42*val30))));
  }
  var alu27 = ((gidx0<<5)+(lidx0<<2));
  var val54 = data3[alu27];
  var val55 = data4[alu27];
  var alu28 = (alu27+1);
  var val56 = data3[alu28];
  var val57 = data4[alu28];
  var alu29 = (alu27+2);
  var val58 = data3[alu29];
  var val59 = data4[alu29];
  var alu30 = (alu27+3);
  var val60 = data3[alu30];
  var val61 = data4[alu30];
  var alu31 = ((gidx0<<11)+alu0+(lidx0<<8)+alu1+alu2);
  data0[alu31] = (val55+(f16(acc0))+val54);
  data0[(alu31+1)] = (val55+(f16(acc4))+val54);
  data0[(alu31+2)] = (val55+(f16(acc8))+val54);
  data0[(alu31+3)] = (val55+(f16(acc12))+val54);
  data0[(alu31+64)] = (val57+(f16(acc1))+val56);
  data0[(alu31+65)] = (val57+(f16(acc5))+val56);
  data0[(alu31+66)] = (val57+(f16(acc9))+val56);
  data0[(alu31+67)] = (val57+(f16(acc13))+val56);
  data0[(alu31+128)] = (val59+(f16(acc2))+val58);
  data0[(alu31+129)] = (val59+(f16(acc6))+val58);
  data0[(alu31+130)] = (val59+(f16(acc10))+val58);
  data0[(alu31+131)] = (val59+(f16(acc14))+val58);
  data0[(alu31+192)] = (val61+(f16(acc3))+val60);
  data0[(alu31+193)] = (val61+(f16(acc7))+val60);
  data0[(alu31+194)] = (val61+(f16(acc11))+val60);
  data0[(alu31+195)] = (val61+(f16(acc15))+val60);
}`;

const r_2_40_8_8_2_1280_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (gidx1*81920);
  var alu1 = (lidx1<<3);
  var alu2 = (lidx2<<2);
  var alu3 = (lidx2<1);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (alu3!=true);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx0*368640)+(lidx0*46080)+(ridx0*9));
    var val0 = data3[alu7];
    var val1 = data3[(alu7+1)];
    var val2 = data3[(alu7+2)];
    var val3 = data3[(alu7+3)];
    var val4 = data3[(alu7+4)];
    var val5 = data3[(alu7+5)];
    var val6 = data3[(alu7+6)];
    var val7 = data3[(alu7+7)];
    var val8 = data3[(alu7+8)];
    var val9 = data3[(alu7+11520)];
    var val10 = data3[(alu7+11521)];
    var val11 = data3[(alu7+11522)];
    var val12 = data3[(alu7+11523)];
    var val13 = data3[(alu7+11524)];
    var val14 = data3[(alu7+11525)];
    var val15 = data3[(alu7+11526)];
    var val16 = data3[(alu7+11527)];
    var val17 = data3[(alu7+11528)];
    var val18 = data3[(alu7+23040)];
    var val19 = data3[(alu7+23041)];
    var val20 = data3[(alu7+23042)];
    var val21 = data3[(alu7+23043)];
    var val22 = data3[(alu7+23044)];
    var val23 = data3[(alu7+23045)];
    var val24 = data3[(alu7+23046)];
    var val25 = data3[(alu7+23047)];
    var val26 = data3[(alu7+23048)];
    var val27 = data3[(alu7+34560)];
    var val28 = data3[(alu7+34561)];
    var val29 = data3[(alu7+34562)];
    var val30 = data3[(alu7+34563)];
    var val31 = data3[(alu7+34564)];
    var val32 = data3[(alu7+34565)];
    var val33 = data3[(alu7+34566)];
    var val34 = data3[(alu7+34567)];
    var val35 = data3[(alu7+34568)];
    var alu8 = (alu0+(ridx0<<6)+alu1);
    var alu9 = (alu8+alu2);
    var val36 = data2[alu9];
    var val37 = select((f16(0.0f)), data2[(alu9+-8)], alu4);
    var val38 = select((f16(0.0f)), data2[(alu9+-7)], alu4);
    var val39 = select((f16(0.0f)), data2[(alu9+-6)], alu4);
    var val40 = select((f16(0.0f)), data2[(alu9+-5)], alu4);
    var val41 = data2[(alu9+1)];
    var val42 = data2[(alu9+2)];
    var val43 = data2[(alu9+3)];
    var val44 = select((f16(0.0f)), data2[(alu9+8)], alu6);
    var val45 = select((f16(0.0f)), data2[(alu9+9)], alu6);
    var val46 = select((f16(0.0f)), data2[(alu9+10)], alu6);
    var val47 = select((f16(0.0f)), data2[(alu9+11)], alu6);
    var val48 = select((f16(0.0f)), data2[(alu8+-5)], (alu4&alu5));
    var val49 = select((f16(0.0f)), data2[(alu8+-4)], (alu3&alu4));
    var val50 = select((f16(0.0f)), data2[(alu8+3)], alu5);
    var val51 = select((f16(0.0f)), data2[(alu8+4)], alu3);
    var val52 = select((f16(0.0f)), data2[(alu8+11)], (alu6&alu5));
    var val53 = select((f16(0.0f)), data2[(alu8+12)], (alu6&alu3));
    acc0 = (acc0+(f32((val45*val8)))+(f32((val41*val5)))+(f32((val38*val2)))+(f32((val44*val7)))+(f32((val36*val4)))+(f32((val37*val1)))+(f32((val52*val6)))+(f32((val48*val0)))+(f32((val50*val3))));
    acc1 = (acc1+(f32((val45*val17)))+(f32((val41*val14)))+(f32((val38*val11)))+(f32((val44*val16)))+(f32((val36*val13)))+(f32((val37*val10)))+(f32((val52*val15)))+(f32((val48*val9)))+(f32((val50*val12))));
    acc2 = (acc2+(f32((val45*val26)))+(f32((val41*val23)))+(f32((val38*val20)))+(f32((val44*val25)))+(f32((val36*val22)))+(f32((val37*val19)))+(f32((val52*val24)))+(f32((val48*val18)))+(f32((val50*val21))));
    acc3 = (acc3+(f32((val45*val35)))+(f32((val41*val32)))+(f32((val38*val29)))+(f32((val44*val34)))+(f32((val36*val31)))+(f32((val37*val28)))+(f32((val52*val33)))+(f32((val48*val27)))+(f32((val50*val30))));
    acc4 = (acc4+(f32((val46*val8)))+(f32((val42*val5)))+(f32((val39*val2)))+(f32((val45*val7)))+(f32((val41*val4)))+(f32((val38*val1)))+(f32((val44*val6)))+(f32((val37*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val46*val17)))+(f32((val42*val14)))+(f32((val39*val11)))+(f32((val45*val16)))+(f32((val41*val13)))+(f32((val38*val10)))+(f32((val44*val15)))+(f32((val37*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val46*val26)))+(f32((val42*val23)))+(f32((val39*val20)))+(f32((val45*val25)))+(f32((val41*val22)))+(f32((val38*val19)))+(f32((val44*val24)))+(f32((val37*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val46*val35)))+(f32((val42*val32)))+(f32((val39*val29)))+(f32((val45*val34)))+(f32((val41*val31)))+(f32((val38*val28)))+(f32((val44*val33)))+(f32((val37*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val47*val8)))+(f32((val43*val5)))+(f32((val40*val2)))+(f32((val46*val7)))+(f32((val42*val4)))+(f32((val39*val1)))+(f32((val45*val6)))+(f32((val38*val0)))+(f32((val41*val3))));
    acc9 = (acc9+(f32((val47*val17)))+(f32((val43*val14)))+(f32((val40*val11)))+(f32((val46*val16)))+(f32((val42*val13)))+(f32((val39*val10)))+(f32((val45*val15)))+(f32((val38*val9)))+(f32((val41*val12))));
    acc10 = (acc10+(f32((val47*val26)))+(f32((val43*val23)))+(f32((val40*val20)))+(f32((val46*val25)))+(f32((val42*val22)))+(f32((val39*val19)))+(f32((val45*val24)))+(f32((val38*val18)))+(f32((val41*val21))));
    acc11 = (acc11+(f32((val47*val35)))+(f32((val43*val32)))+(f32((val40*val29)))+(f32((val46*val34)))+(f32((val42*val31)))+(f32((val39*val28)))+(f32((val45*val33)))+(f32((val38*val27)))+(f32((val41*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val51*val5)))+(f32((val49*val2)))+(f32((val47*val7)))+(f32((val43*val4)))+(f32((val40*val1)))+(f32((val46*val6)))+(f32((val39*val0)))+(f32((val42*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val51*val14)))+(f32((val49*val11)))+(f32((val47*val16)))+(f32((val43*val13)))+(f32((val40*val10)))+(f32((val46*val15)))+(f32((val39*val9)))+(f32((val42*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val51*val23)))+(f32((val49*val20)))+(f32((val47*val25)))+(f32((val43*val22)))+(f32((val40*val19)))+(f32((val46*val24)))+(f32((val39*val18)))+(f32((val42*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val51*val32)))+(f32((val49*val29)))+(f32((val47*val34)))+(f32((val43*val31)))+(f32((val40*val28)))+(f32((val46*val33)))+(f32((val39*val27)))+(f32((val42*val30))));
  }
  var alu27 = ((gidx0<<5)+(lidx0<<2));
  var val54 = data4[alu27];
  var val55 = data4[(alu27+1)];
  var val56 = data4[(alu27+2)];
  var val57 = data4[(alu27+3)];
  var alu28 = ((gidx0<<11)+alu0+(lidx0<<8)+alu1+alu2);
  var val58 = data1[alu28];
  var alu29 = (alu28+1);
  var val59 = data1[alu29];
  var alu30 = (alu28+2);
  var val60 = data1[alu30];
  var alu31 = (alu28+3);
  var val61 = data1[alu31];
  var alu32 = (alu28+64);
  var val62 = data1[alu32];
  var alu33 = (alu28+65);
  var val63 = data1[alu33];
  var alu34 = (alu28+66);
  var val64 = data1[alu34];
  var alu35 = (alu28+67);
  var val65 = data1[alu35];
  var alu36 = (alu28+128);
  var val66 = data1[alu36];
  var alu37 = (alu28+129);
  var val67 = data1[alu37];
  var alu38 = (alu28+130);
  var val68 = data1[alu38];
  var alu39 = (alu28+131);
  var val69 = data1[alu39];
  var alu40 = (alu28+192);
  var val70 = data1[alu40];
  var alu41 = (alu28+193);
  var val71 = data1[alu41];
  var alu42 = (alu28+194);
  var val72 = data1[alu42];
  var alu43 = (alu28+195);
  var val73 = data1[alu43];
  data0[alu29] = (val59+(f16(acc4))+val54);
  data0[alu30] = (val60+(f16(acc8))+val54);
  data0[alu31] = (val61+(f16(acc12))+val54);
  data0[alu32] = (val62+(f16(acc1))+val55);
  data0[alu33] = (val63+(f16(acc5))+val55);
  data0[alu34] = (val64+(f16(acc9))+val55);
  data0[alu35] = (val65+(f16(acc13))+val55);
  data0[alu36] = (val66+(f16(acc2))+val56);
  data0[alu37] = (val67+(f16(acc6))+val56);
  data0[alu38] = (val68+(f16(acc10))+val56);
  data0[alu39] = (val69+(f16(acc14))+val56);
  data0[alu40] = (val70+(f16(acc3))+val57);
  data0[alu41] = (val71+(f16(acc7))+val57);
  data0[alu42] = (val72+(f16(acc11))+val57);
  data0[alu43] = (val73+(f16(acc15))+val57);
  data0[alu28] = (val58+(f16(acc0))+val54);
}`;

const E_2_160_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx0<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx1<<5)+(gidx0/5));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx0<<9)+(gidx1*81920)+(lidx0<<6)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  data0[alu3] = (val1+(val0*val3*(val5-val2)));
  data0[alu4] = (val1+(val0*val3*(val6-val2)));
  data0[alu5] = (val1+(val0*val3*(val7-val2)));
  data0[alu2] = (val1+(val0*val3*(val4-val2)));
}`;

const r_2_40_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1*81920);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu2 = ((gidx0*40960)+(lidx0*5120)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+1280)];
    var val5 = data2[(alu2+1281)];
    var val6 = data2[(alu2+1282)];
    var val7 = data2[(alu2+1283)];
    var val8 = data2[(alu2+2560)];
    var val9 = data2[(alu2+2561)];
    var val10 = data2[(alu2+2562)];
    var val11 = data2[(alu2+2563)];
    var val12 = data2[(alu2+3840)];
    var val13 = data2[(alu2+3841)];
    var val14 = data2[(alu2+3842)];
    var val15 = data2[(alu2+3843)];
    var alu3 = (alu0+alu1+(ridx0<<8));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+64)];
    var val21 = data1[(alu3+65)];
    var val22 = data1[(alu3+66)];
    var val23 = data1[(alu3+67)];
    var val24 = data1[(alu3+128)];
    var val25 = data1[(alu3+129)];
    var val26 = data1[(alu3+130)];
    var val27 = data1[(alu3+131)];
    var val28 = data1[(alu3+192)];
    var val29 = data1[(alu3+193)];
    var val30 = data1[(alu3+194)];
    var val31 = data1[(alu3+195)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
  }
  var alu21 = ((gidx0<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data3[(alu21+1)];
  var val34 = data3[(alu21+2)];
  var val35 = data3[(alu21+3)];
  var alu22 = ((gidx0<<11)+alu0+(lidx0<<8)+alu1);
  data0[alu22] = ((f16(acc0))+val32);
  data0[(alu22+1)] = ((f16(acc4))+val32);
  data0[(alu22+2)] = ((f16(acc8))+val32);
  data0[(alu22+3)] = ((f16(acc12))+val32);
  data0[(alu22+64)] = ((f16(acc1))+val33);
  data0[(alu22+65)] = ((f16(acc5))+val33);
  data0[(alu22+66)] = ((f16(acc9))+val33);
  data0[(alu22+67)] = ((f16(acc13))+val33);
  data0[(alu22+128)] = ((f16(acc2))+val34);
  data0[(alu22+129)] = ((f16(acc6))+val34);
  data0[(alu22+130)] = ((f16(acc10))+val34);
  data0[(alu22+131)] = ((f16(acc14))+val34);
  data0[(alu22+192)] = ((f16(acc3))+val35);
  data0[(alu22+193)] = ((f16(acc7))+val35);
  data0[(alu22+194)] = ((f16(acc11))+val35);
  data0[(alu22+195)] = ((f16(acc15))+val35);
}`;

const r_2_64_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val0 = data1[(gidx0+(gidx1*81920)+(lidx0*5120)+(ridx0<<6))];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[(gidx0+(gidx1<<6))] = (f16((acc1*0.0007812500116415322f)));
  }
}`;

const r_2_64_16_80n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0+(gidx1<<6));
  var val0 = data2[alu0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data1[(gidx0+(gidx1*81920)+(lidx0*5120)+(ridx0<<6))];
    var alu1 = (val1-val0);
    acc0 = (acc0+(f32((alu1*alu1))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[alu0] = sqrt((1/((f16((acc1*0.0007812500116415322f)))+(f16(1e-05f)))));
  }
}`;

const E_2_2_20_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*81920);
  var alu1 = (gidx0<<6);
  var alu2 = (gidx1<<5);
  var alu3 = (lidx0<<2);
  var alu4 = (alu2+(gidx2<<6)+alu3);
  var val0 = data2[alu4];
  var val1 = data3[alu4];
  var alu5 = (alu4+1);
  var val2 = data2[alu5];
  var val3 = data3[alu5];
  var alu6 = (alu4+2);
  var val4 = data2[alu6];
  var val5 = data3[alu6];
  var alu7 = (alu4+3);
  var val6 = data2[alu7];
  var val7 = data3[alu7];
  var alu8 = (lidx1<<2);
  var alu9 = (alu1+alu8);
  var val8 = data4[alu9];
  var val9 = data5[alu9];
  var alu10 = (alu9+1);
  var val10 = data4[alu10];
  var val11 = data5[alu10];
  var alu11 = (alu9+2);
  var val12 = data4[alu11];
  var val13 = data5[alu11];
  var alu12 = (alu9+3);
  var val14 = data4[alu12];
  var val15 = data5[alu12];
  var alu13 = (alu2+alu0+(gidx0<<12)+alu3+(lidx1<<8));
  var val16 = data1[alu13];
  var val17 = data1[(alu13+1)];
  var val18 = data1[(alu13+2)];
  var val19 = data1[(alu13+3)];
  var val20 = data1[(alu13+64)];
  var val21 = data1[(alu13+65)];
  var val22 = data1[(alu13+66)];
  var val23 = data1[(alu13+67)];
  var val24 = data1[(alu13+128)];
  var val25 = data1[(alu13+129)];
  var val26 = data1[(alu13+130)];
  var val27 = data1[(alu13+131)];
  var val28 = data1[(alu13+192)];
  var val29 = data1[(alu13+193)];
  var val30 = data1[(alu13+194)];
  var val31 = data1[(alu13+195)];
  var alu14 = ((gidx1*40960)+alu0+alu1+(lidx0*5120)+alu8);
  data0[(alu14+1280)] = (val9+(val8*val3*(val17-val2)));
  data0[(alu14+2560)] = (val9+(val8*val5*(val18-val4)));
  data0[(alu14+3840)] = (val9+(val8*val7*(val19-val6)));
  data0[(alu14+1)] = (val11+(val10*val1*(val20-val0)));
  data0[(alu14+1281)] = (val11+(val10*val3*(val21-val2)));
  data0[(alu14+2561)] = (val11+(val10*val5*(val22-val4)));
  data0[(alu14+3841)] = (val11+(val10*val7*(val23-val6)));
  data0[(alu14+2)] = (val13+(val12*val1*(val24-val0)));
  data0[(alu14+1282)] = (val13+(val12*val3*(val25-val2)));
  data0[(alu14+2562)] = (val13+(val12*val5*(val26-val4)));
  data0[(alu14+3842)] = (val13+(val12*val7*(val27-val6)));
  data0[(alu14+3)] = (val15+(val14*val1*(val28-val0)));
  data0[(alu14+1283)] = (val15+(val14*val3*(val29-val2)));
  data0[(alu14+2563)] = (val15+(val14*val5*(val30-val4)));
  data0[(alu14+3843)] = (val15+(val14*val7*(val31-val6)));
  data0[alu14] = (val9+(val8*val1*(val16-val0)));
}`;

const r_4_20_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1*40960);
  var alu1 = (lidx0*5120);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu2 = (ridx0<<2);
    var alu3 = ((gidx0*81920)+(lidx1*5120)+alu2);
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+1280)];
    var val5 = data2[(alu3+1281)];
    var val6 = data2[(alu3+1282)];
    var val7 = data2[(alu3+1283)];
    var val8 = data2[(alu3+2560)];
    var val9 = data2[(alu3+2561)];
    var val10 = data2[(alu3+2562)];
    var val11 = data2[(alu3+2563)];
    var val12 = data2[(alu3+3840)];
    var val13 = data2[(alu3+3841)];
    var val14 = data2[(alu3+3842)];
    var val15 = data2[(alu3+3843)];
    var alu4 = (alu0+alu1+alu2);
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+1280)];
    var val21 = data1[(alu4+1281)];
    var val22 = data1[(alu4+1282)];
    var val23 = data1[(alu4+1283)];
    var val24 = data1[(alu4+2560)];
    var val25 = data1[(alu4+2561)];
    var val26 = data1[(alu4+2562)];
    var val27 = data1[(alu4+2563)];
    var val28 = data1[(alu4+3840)];
    var val29 = data1[(alu4+3841)];
    var val30 = data1[(alu4+3842)];
    var val31 = data1[(alu4+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu22 = ((gidx0<<6)+alu0+alu1+(lidx1<<2));
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+1280)] = (f16(acc1));
  data0[(alu22+1281)] = (f16(acc5));
  data0[(alu22+1282)] = (f16(acc9));
  data0[(alu22+1283)] = (f16(acc13));
  data0[(alu22+2560)] = (f16(acc2));
  data0[(alu22+2561)] = (f16(acc6));
  data0[(alu22+2562)] = (f16(acc10));
  data0[(alu22+2563)] = (f16(acc14));
  data0[(alu22+3840)] = (f16(acc3));
  data0[(alu22+3841)] = (f16(acc7));
  data0[(alu22+3842)] = (f16(acc11));
  data0[(alu22+3843)] = (f16(acc15));
}`;

const r_2_8_2_8_16_40_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = ((gidx1*160)+(gidx2*81920));
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
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = (alu0+(gidx0*40960)+(lidx0*5120)+alu1);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+1)];
    var val2 = data1[(alu2+2)];
    var val3 = data1[(alu2+3)];
    var val4 = data1[(alu2+1280)];
    var val5 = data1[(alu2+1281)];
    var val6 = data1[(alu2+1282)];
    var val7 = data1[(alu2+1283)];
    var val8 = data1[(alu2+2560)];
    var val9 = data1[(alu2+2561)];
    var val10 = data1[(alu2+2562)];
    var val11 = data1[(alu2+2563)];
    var val12 = data1[(alu2+3840)];
    var val13 = data1[(alu2+3841)];
    var val14 = data1[(alu2+3842)];
    var val15 = data1[(alu2+3843)];
    var alu3 = (alu0+(lidx1*5120)+alu1);
    var val16 = data2[alu3];
    var val17 = data2[(alu3+1)];
    var val18 = data2[(alu3+2)];
    var val19 = data2[(alu3+3)];
    var val20 = data2[(alu3+1280)];
    var val21 = data2[(alu3+1281)];
    var val22 = data2[(alu3+1282)];
    var val23 = data2[(alu3+1283)];
    var val24 = data2[(alu3+2560)];
    var val25 = data2[(alu3+2561)];
    var val26 = data2[(alu3+2562)];
    var val27 = data2[(alu3+2563)];
    var val28 = data2[(alu3+3840)];
    var val29 = data2[(alu3+3841)];
    var val30 = data2[(alu3+3842)];
    var val31 = data2[(alu3+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc2 = (acc2+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc3 = (acc3+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc4 = (acc4+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc7 = (acc7+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc8 = (acc8+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc9 = (acc9+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc12 = (acc12+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc13 = (acc13+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc14 = (acc14+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu21 = ((gidx1<<12)+(gidx2<<15)+(gidx0<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu21] = (acc0*0.07905694097280502f);
  data0[(alu21+1)] = (acc4*0.07905694097280502f);
  data0[(alu21+2)] = (acc8*0.07905694097280502f);
  data0[(alu21+3)] = (acc12*0.07905694097280502f);
  data0[(alu21+64)] = (acc1*0.07905694097280502f);
  data0[(alu21+65)] = (acc5*0.07905694097280502f);
  data0[(alu21+66)] = (acc9*0.07905694097280502f);
  data0[(alu21+67)] = (acc13*0.07905694097280502f);
  data0[(alu21+128)] = (acc2*0.07905694097280502f);
  data0[(alu21+129)] = (acc6*0.07905694097280502f);
  data0[(alu21+130)] = (acc10*0.07905694097280502f);
  data0[(alu21+131)] = (acc14*0.07905694097280502f);
  data0[(alu21+192)] = (acc3*0.07905694097280502f);
  data0[(alu21+193)] = (acc7*0.07905694097280502f);
  data0[(alu21+194)] = (acc11*0.07905694097280502f);
  data0[(alu21+195)] = (acc15*0.07905694097280502f);
}`;

const r_1024_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 4; ridx0++) {
    var val0 = data1[((gidx0<<6)+(lidx0<<2)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = (f32(-INFINITY));
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = select(acc1,val1,(acc1<val1));
    }
    data0[gidx0] = acc1;
  }
}`;

const r_1024_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 4; ridx0++) {
    var val1 = data1[((gidx0<<6)+(lidx0<<2)+ridx0)];
    acc0 = (acc0+exp2(((val1-val0)*1.4426950408889634f)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = acc1;
  }
}`;

const E_128_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx0<<3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var alu1 = ((gidx0<<9)+(lidx0<<6)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (1/val1);
  data0[alu2] = (f16((exp2(((val3-val0)*1.4426950408889634f))*alu5)));
  data0[alu3] = (f16((exp2(((val4-val0)*1.4426950408889634f))*alu5)));
  data0[alu4] = (f16((exp2(((val5-val0)*1.4426950408889634f))*alu5)));
  data0[alu1] = (f16((exp2(((val2-val0)*1.4426950408889634f))*alu5)));
}`;

const r_2_8_5_16_8_16_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx2*81920);
  var alu1 = (gidx0<<5);
  var alu2 = (lidx1<<2);
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
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var alu3 = ((gidx1*160)+alu0+alu1+alu2+(ridx0*5120));
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+1280)];
    var val5 = data2[(alu3+1281)];
    var val6 = data2[(alu3+1282)];
    var val7 = data2[(alu3+1283)];
    var val8 = data2[(alu3+2560)];
    var val9 = data2[(alu3+2561)];
    var val10 = data2[(alu3+2562)];
    var val11 = data2[(alu3+2563)];
    var val12 = data2[(alu3+3840)];
    var val13 = data2[(alu3+3841)];
    var val14 = data2[(alu3+3842)];
    var val15 = data2[(alu3+3843)];
    var alu4 = ((gidx1<<12)+(gidx2<<15)+(lidx0<<8)+(ridx0<<2));
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+64)];
    var val21 = data1[(alu4+65)];
    var val22 = data1[(alu4+66)];
    var val23 = data1[(alu4+67)];
    var val24 = data1[(alu4+128)];
    var val25 = data1[(alu4+129)];
    var val26 = data1[(alu4+130)];
    var val27 = data1[(alu4+131)];
    var val28 = data1[(alu4+192)];
    var val29 = data1[(alu4+193)];
    var val30 = data1[(alu4+194)];
    var val31 = data1[(alu4+195)];
    acc0 = (acc0+(f32((val12*val19)))+(f32((val8*val18)))+(f32((val4*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val12*val23)))+(f32((val8*val22)))+(f32((val4*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val12*val27)))+(f32((val8*val26)))+(f32((val4*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val12*val31)))+(f32((val8*val30)))+(f32((val4*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val13*val19)))+(f32((val9*val18)))+(f32((val1*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val13*val23)))+(f32((val9*val22)))+(f32((val1*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val13*val27)))+(f32((val9*val26)))+(f32((val1*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val13*val31)))+(f32((val9*val30)))+(f32((val1*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val14*val19)))+(f32((val10*val18)))+(f32((val2*val16)))+(f32((val6*val17))));
    acc9 = (acc9+(f32((val14*val23)))+(f32((val10*val22)))+(f32((val2*val20)))+(f32((val6*val21))));
    acc10 = (acc10+(f32((val14*val27)))+(f32((val10*val26)))+(f32((val2*val24)))+(f32((val6*val25))));
    acc11 = (acc11+(f32((val14*val31)))+(f32((val10*val30)))+(f32((val2*val28)))+(f32((val6*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val11*val18)))+(f32((val3*val16)))+(f32((val7*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val11*val22)))+(f32((val3*val20)))+(f32((val7*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val11*val26)))+(f32((val3*val24)))+(f32((val7*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val11*val30)))+(f32((val3*val28)))+(f32((val7*val29))));
  }
  var alu22 = ((gidx1*10240)+alu0+alu1+(lidx0*640)+alu2);
  data0[alu22] = (f16(acc0));
  data0[(alu22+1)] = (f16(acc4));
  data0[(alu22+2)] = (f16(acc8));
  data0[(alu22+3)] = (f16(acc12));
  data0[(alu22+160)] = (f16(acc1));
  data0[(alu22+161)] = (f16(acc5));
  data0[(alu22+162)] = (f16(acc9));
  data0[(alu22+163)] = (f16(acc13));
  data0[(alu22+320)] = (f16(acc2));
  data0[(alu22+321)] = (f16(acc6));
  data0[(alu22+322)] = (f16(acc10));
  data0[(alu22+323)] = (f16(acc14));
  data0[(alu22+480)] = (f16(acc3));
  data0[(alu22+481)] = (f16(acc7));
  data0[(alu22+482)] = (f16(acc11));
  data0[(alu22+483)] = (f16(acc15));
}`;

const r_2_2_20_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*81920);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu1 = ((gidx1*5120)+(lidx0*640)+alu0+((ridx0/40)*10240)+((ridx0%40)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+160)];
    var val5 = data2[(alu1+161)];
    var val6 = data2[(alu1+162)];
    var val7 = data2[(alu1+163)];
    var val8 = data2[(alu1+320)];
    var val9 = data2[(alu1+321)];
    var val10 = data2[(alu1+322)];
    var val11 = data2[(alu1+323)];
    var val12 = data2[(alu1+480)];
    var val13 = data2[(alu1+481)];
    var val14 = data2[(alu1+482)];
    var val15 = data2[(alu1+483)];
    var alu2 = ((gidx0*81920)+(lidx1*5120)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+1280)];
    var val21 = data3[(alu2+1281)];
    var val22 = data3[(alu2+1282)];
    var val23 = data3[(alu2+1283)];
    var val24 = data3[(alu2+2560)];
    var val25 = data3[(alu2+2561)];
    var val26 = data3[(alu2+2562)];
    var val27 = data3[(alu2+2563)];
    var val28 = data3[(alu2+3840)];
    var val29 = data3[(alu2+3841)];
    var val30 = data3[(alu2+3842)];
    var val31 = data3[(alu2+3843)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data4[alu22];
  var val33 = data4[(alu22+1)];
  var val34 = data4[(alu22+2)];
  var val35 = data4[(alu22+3)];
  var alu23 = ((gidx1<<5)+alu0+(gidx0<<12)+(lidx0<<2)+(lidx1<<8));
  var val36 = data1[alu23];
  var val37 = data1[(alu23+1)];
  var val38 = data1[(alu23+2)];
  var val39 = data1[(alu23+3)];
  var val40 = data1[(alu23+64)];
  var val41 = data1[(alu23+65)];
  var val42 = data1[(alu23+66)];
  var val43 = data1[(alu23+67)];
  var val44 = data1[(alu23+128)];
  var val45 = data1[(alu23+129)];
  var val46 = data1[(alu23+130)];
  var val47 = data1[(alu23+131)];
  var val48 = data1[(alu23+192)];
  var val49 = data1[(alu23+193)];
  var val50 = data1[(alu23+194)];
  var val51 = data1[(alu23+195)];
  var alu24 = ((gidx1*40960)+alu0+alu20+(lidx0*5120)+alu21);
  data0[alu24] = (val36+(f16(acc0))+val32);
  data0[(alu24+1)] = (val40+(f16(acc4))+val33);
  data0[(alu24+2)] = (val44+(f16(acc8))+val34);
  data0[(alu24+3)] = (val48+(f16(acc12))+val35);
  data0[(alu24+1280)] = (val37+(f16(acc1))+val32);
  data0[(alu24+1281)] = (val41+(f16(acc5))+val33);
  data0[(alu24+1282)] = (val45+(f16(acc9))+val34);
  data0[(alu24+1283)] = (val49+(f16(acc13))+val35);
  data0[(alu24+2560)] = (val38+(f16(acc2))+val32);
  data0[(alu24+2561)] = (val42+(f16(acc6))+val33);
  data0[(alu24+2562)] = (val46+(f16(acc10))+val34);
  data0[(alu24+2563)] = (val50+(f16(acc14))+val35);
  data0[(alu24+3840)] = (val39+(f16(acc3))+val32);
  data0[(alu24+3841)] = (val43+(f16(acc7))+val33);
  data0[(alu24+3842)] = (val47+(f16(acc11))+val34);
  data0[(alu24+3843)] = (val51+(f16(acc15))+val35);
}`;

const r_128_16_80 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val0 = data1[((gidx0*1280)+(lidx0*80)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*0.0007812500116415322f)));
  }
}`;

const r_128_16_80n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var val1 = data1[((gidx0*1280)+(lidx0*80)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*0.0007812500116415322f)))+(f16(1e-05f)))));
  }
}`;

const E_4_20_8_16_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data2[alu1];
  var val1 = data3[alu1];
  var alu2 = (alu1+1);
  var val2 = data2[alu2];
  var val3 = data3[alu2];
  var alu3 = (alu1+2);
  var val4 = data2[alu3];
  var val5 = data3[alu3];
  var alu4 = (alu1+3);
  var val6 = data2[alu4];
  var val7 = data3[alu4];
  var alu5 = (lidx1<<2);
  var alu6 = (alu0+(gidx1*40960)+(lidx0*5120)+alu5);
  var val8 = data1[alu6];
  var alu7 = (alu6+1);
  var val9 = data1[alu7];
  var alu8 = (alu6+2);
  var val10 = data1[alu8];
  var alu9 = (alu6+3);
  var val11 = data1[alu9];
  var alu10 = (alu6+1280);
  var val12 = data1[alu10];
  var alu11 = (alu6+1281);
  var val13 = data1[alu11];
  var alu12 = (alu6+1282);
  var val14 = data1[alu12];
  var alu13 = (alu6+1283);
  var val15 = data1[alu13];
  var alu14 = (alu6+2560);
  var val16 = data1[alu14];
  var alu15 = (alu6+2561);
  var val17 = data1[alu15];
  var alu16 = (alu6+2562);
  var val18 = data1[alu16];
  var alu17 = (alu6+2563);
  var val19 = data1[alu17];
  var alu18 = (alu6+3840);
  var val20 = data1[alu18];
  var alu19 = (alu6+3841);
  var val21 = data1[alu19];
  var alu20 = (alu6+3842);
  var val22 = data1[alu20];
  var alu21 = (alu6+3843);
  var val23 = data1[alu21];
  var alu22 = (alu0+alu5);
  var val24 = data4[alu22];
  var val25 = data5[alu22];
  var alu23 = (alu22+1);
  var val26 = data4[alu23];
  var val27 = data5[alu23];
  var alu24 = (alu22+2);
  var val28 = data4[alu24];
  var val29 = data5[alu24];
  var alu25 = (alu22+3);
  var val30 = data4[alu25];
  var val31 = data5[alu25];
  data0[alu7] = (val27+(val26*val1*(val9-val0)));
  data0[alu8] = (val29+(val28*val1*(val10-val0)));
  data0[alu9] = (val31+(val30*val1*(val11-val0)));
  data0[alu10] = (val25+(val24*val3*(val12-val2)));
  data0[alu11] = (val27+(val26*val3*(val13-val2)));
  data0[alu12] = (val29+(val28*val3*(val14-val2)));
  data0[alu13] = (val31+(val30*val3*(val15-val2)));
  data0[alu14] = (val25+(val24*val5*(val16-val4)));
  data0[alu15] = (val27+(val26*val5*(val17-val4)));
  data0[alu16] = (val29+(val28*val5*(val18-val4)));
  data0[alu17] = (val31+(val30*val5*(val19-val4)));
  data0[alu18] = (val25+(val24*val7*(val20-val6)));
  data0[alu19] = (val27+(val26*val7*(val21-val6)));
  data0[alu20] = (val29+(val28*val7*(val22-val6)));
  data0[alu21] = (val31+(val30*val7*(val23-val6)));
  data0[alu6] = (val25+(val24*val1*(val8-val0)));
}`;

const r_2_77_8_16_40_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0*160);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 40; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*1280)+(gidx1*98560)+alu0+alu1);
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var alu3 = ((gidx1*81920)+alu0+(lidx1*5120)+alu1);
    var val4 = data1[alu3];
    var val5 = data1[(alu3+1)];
    var val6 = data1[(alu3+2)];
    var val7 = data1[(alu3+3)];
    var val8 = data1[(alu3+1280)];
    var val9 = data1[(alu3+1281)];
    var val10 = data1[(alu3+1282)];
    var val11 = data1[(alu3+1283)];
    var val12 = data1[(alu3+2560)];
    var val13 = data1[(alu3+2561)];
    var val14 = data1[(alu3+2562)];
    var val15 = data1[(alu3+2563)];
    var val16 = data1[(alu3+3840)];
    var val17 = data1[(alu3+3841)];
    var val18 = data1[(alu3+3842)];
    var val19 = data1[(alu3+3843)];
    acc0 = (acc0+(f32((val3*val7)))+(f32((val2*val6)))+(f32((val1*val5)))+(f32((val0*val4))));
    acc1 = (acc1+(f32((val3*val11)))+(f32((val2*val10)))+(f32((val1*val9)))+(f32((val0*val8))));
    acc2 = (acc2+(f32((val3*val15)))+(f32((val2*val14)))+(f32((val1*val13)))+(f32((val0*val12))));
    acc3 = (acc3+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
  }
  var alu9 = (gidx0+(gidx1*39424)+(lidx0*4928)+(lidx1*308));
  data0[alu9] = (acc0*0.07905694097280502f);
  data0[(alu9+77)] = (acc1*0.07905694097280502f);
  data0[(alu9+154)] = (acc2*0.07905694097280502f);
  data0[(alu9+231)] = (acc3*0.07905694097280502f);
}`;

const r_32_32_77 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-INFINITY));
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var val0 = data1[((gidx0*2464)+(lidx0*77)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_8_32_77_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu4 = ((gidx0*9856)+(lidx0*308)+ridx0);
    var val4 = data1[alu4];
    var val5 = data1[(alu4+77)];
    var val6 = data1[(alu4+154)];
    var val7 = data1[(alu4+231)];
    acc0 = (acc0+exp2(((val4-val3)*1.4426950408889634f)));
    acc1 = (acc1+exp2(((val5-val0)*1.4426950408889634f)));
    acc2 = (acc2+exp2(((val6-val1)*1.4426950408889634f)));
    acc3 = (acc3+exp2(((val7-val2)*1.4426950408889634f)));
  }
  data0[alu1] = acc1;
  data0[alu2] = acc2;
  data0[alu3] = acc3;
  data0[alu0] = acc0;
}`;

const E_8_77_32_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 77 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (gidx0+(gidx1*9856)+(lidx0*308));
  var val0 = data1[alu0];
  var alu1 = (alu0+77);
  var val1 = data1[alu1];
  var alu2 = (alu0+154);
  var val2 = data1[alu2];
  var alu3 = (alu0+231);
  var val3 = data1[alu3];
  var alu4 = ((gidx1<<7)+(lidx0<<2));
  var val4 = data2[alu4];
  var val5 = data3[alu4];
  var alu5 = (alu4+1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = (alu4+2);
  var val8 = data2[alu6];
  var val9 = data3[alu6];
  var alu7 = (alu4+3);
  var val10 = data2[alu7];
  var val11 = data3[alu7];
  data0[alu0] = (f16((exp2(((val0-val4)*1.4426950408889634f))*(1/val5))));
  data0[alu1] = (f16((exp2(((val1-val6)*1.4426950408889634f))*(1/val7))));
  data0[alu2] = (f16((exp2(((val2-val8)*1.4426950408889634f))*(1/val9))));
  data0[alu3] = (f16((exp2(((val3-val10)*1.4426950408889634f))*(1/val11))));
}`;

const r_2_8_5_16_8_77_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx0<<5);
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
  for (var ridx0 = 0; ridx0 < 77; ridx0++) {
    var alu2 = ((gidx1*4928)+(gidx2*39424)+(lidx0*308)+ridx0);
    var val0 = data1[alu2];
    var val1 = data1[(alu2+77)];
    var val2 = data1[(alu2+154)];
    var val3 = data1[(alu2+231)];
    var alu3 = ((gidx1*160)+(gidx2*98560)+alu0+alu1+(ridx0*1280));
    var val4 = data2[alu3];
    var val5 = data2[(alu3+1)];
    var val6 = data2[(alu3+2)];
    var val7 = data2[(alu3+3)];
    acc0 = (acc0+(f32((val4*val0))));
    acc1 = (acc1+(f32((val4*val1))));
    acc2 = (acc2+(f32((val4*val2))));
    acc3 = (acc3+(f32((val4*val3))));
    acc4 = (acc4+(f32((val5*val0))));
    acc5 = (acc5+(f32((val5*val1))));
    acc6 = (acc6+(f32((val5*val2))));
    acc7 = (acc7+(f32((val5*val3))));
    acc8 = (acc8+(f32((val6*val0))));
    acc9 = (acc9+(f32((val6*val1))));
    acc10 = (acc10+(f32((val6*val2))));
    acc11 = (acc11+(f32((val6*val3))));
    acc12 = (acc12+(f32((val7*val0))));
    acc13 = (acc13+(f32((val7*val1))));
    acc14 = (acc14+(f32((val7*val2))));
    acc15 = (acc15+(f32((val7*val3))));
  }
  var alu21 = ((gidx1*10240)+(gidx2*81920)+alu0+(lidx0*640)+alu1);
  data0[alu21] = (f16(acc0));
  data0[(alu21+1)] = (f16(acc4));
  data0[(alu21+2)] = (f16(acc8));
  data0[(alu21+3)] = (f16(acc12));
  data0[(alu21+160)] = (f16(acc1));
  data0[(alu21+161)] = (f16(acc5));
  data0[(alu21+162)] = (f16(acc9));
  data0[(alu21+163)] = (f16(acc13));
  data0[(alu21+320)] = (f16(acc2));
  data0[(alu21+321)] = (f16(acc6));
  data0[(alu21+322)] = (f16(acc10));
  data0[(alu21+323)] = (f16(acc14));
  data0[(alu21+480)] = (f16(acc3));
  data0[(alu21+481)] = (f16(acc7));
  data0[(alu21+482)] = (f16(acc11));
  data0[(alu21+483)] = (f16(acc15));
}`;

const r_2_2_20_8_16_320_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*81920);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu1 = ((gidx1*5120)+(lidx0*640)+alu0+((ridx0/40)*10240)+((ridx0%40)<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+160)];
    var val5 = data2[(alu1+161)];
    var val6 = data2[(alu1+162)];
    var val7 = data2[(alu1+163)];
    var val8 = data2[(alu1+320)];
    var val9 = data2[(alu1+321)];
    var val10 = data2[(alu1+322)];
    var val11 = data2[(alu1+323)];
    var val12 = data2[(alu1+480)];
    var val13 = data2[(alu1+481)];
    var val14 = data2[(alu1+482)];
    var val15 = data2[(alu1+483)];
    var alu2 = ((gidx0*81920)+(lidx1*5120)+(ridx0<<2));
    var val16 = data3[alu2];
    var val17 = data3[(alu2+1)];
    var val18 = data3[(alu2+2)];
    var val19 = data3[(alu2+3)];
    var val20 = data3[(alu2+1280)];
    var val21 = data3[(alu2+1281)];
    var val22 = data3[(alu2+1282)];
    var val23 = data3[(alu2+1283)];
    var val24 = data3[(alu2+2560)];
    var val25 = data3[(alu2+2561)];
    var val26 = data3[(alu2+2562)];
    var val27 = data3[(alu2+2563)];
    var val28 = data3[(alu2+3840)];
    var val29 = data3[(alu2+3841)];
    var val30 = data3[(alu2+3842)];
    var val31 = data3[(alu2+3843)];
    acc0 = (acc0+(f32((val19*val3)))+(f32((val18*val2)))+(f32((val17*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val19*val7)))+(f32((val18*val6)))+(f32((val17*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val19*val11)))+(f32((val18*val10)))+(f32((val17*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val19*val15)))+(f32((val18*val14)))+(f32((val17*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val23*val3)))+(f32((val22*val2)))+(f32((val20*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val23*val7)))+(f32((val22*val6)))+(f32((val20*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val23*val11)))+(f32((val22*val10)))+(f32((val20*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val23*val15)))+(f32((val22*val14)))+(f32((val20*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val27*val3)))+(f32((val26*val2)))+(f32((val24*val0)))+(f32((val25*val1))));
    acc9 = (acc9+(f32((val27*val7)))+(f32((val26*val6)))+(f32((val24*val4)))+(f32((val25*val5))));
    acc10 = (acc10+(f32((val27*val11)))+(f32((val26*val10)))+(f32((val24*val8)))+(f32((val25*val9))));
    acc11 = (acc11+(f32((val27*val15)))+(f32((val26*val14)))+(f32((val24*val12)))+(f32((val25*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val30*val2)))+(f32((val28*val0)))+(f32((val29*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val30*val6)))+(f32((val28*val4)))+(f32((val29*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val30*val10)))+(f32((val28*val8)))+(f32((val29*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val30*val14)))+(f32((val28*val12)))+(f32((val29*val13))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = ((gidx1*40960)+alu0+alu20+(lidx0*5120)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+1280);
  var val36 = data1[alu26];
  var alu27 = (alu22+1281);
  var val37 = data1[alu27];
  var alu28 = (alu22+1282);
  var val38 = data1[alu28];
  var alu29 = (alu22+1283);
  var val39 = data1[alu29];
  var alu30 = (alu22+2560);
  var val40 = data1[alu30];
  var alu31 = (alu22+2561);
  var val41 = data1[alu31];
  var alu32 = (alu22+2562);
  var val42 = data1[alu32];
  var alu33 = (alu22+2563);
  var val43 = data1[alu33];
  var alu34 = (alu22+3840);
  var val44 = data1[alu34];
  var alu35 = (alu22+3841);
  var val45 = data1[alu35];
  var alu36 = (alu22+3842);
  var val46 = data1[alu36];
  var alu37 = (alu22+3843);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_4_160_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 4 */
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*81920)+(lidx1*5120)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+1280)];
    var val5 = data2[(alu1+1281)];
    var val6 = data2[(alu1+1282)];
    var val7 = data2[(alu1+1283)];
    var val8 = data2[(alu1+2560)];
    var val9 = data2[(alu1+2561)];
    var val10 = data2[(alu1+2562)];
    var val11 = data2[(alu1+2563)];
    var val12 = data2[(alu1+3840)];
    var val13 = data2[(alu1+3841)];
    var val14 = data2[(alu1+3842)];
    var val15 = data2[(alu1+3843)];
    var alu2 = ((gidx1*40960)+(lidx0*5120)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+1280)];
    var val21 = data1[(alu2+1281)];
    var val22 = data1[(alu2+1282)];
    var val23 = data1[(alu2+1283)];
    var val24 = data1[(alu2+2560)];
    var val25 = data1[(alu2+2561)];
    var val26 = data1[(alu2+2562)];
    var val27 = data1[(alu2+2563)];
    var val28 = data1[(alu2+3840)];
    var val29 = data1[(alu2+3841)];
    var val30 = data1[(alu2+3842)];
    var val31 = data1[(alu2+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+alu21);
  var val32 = data3[alu22];
  var val33 = data3[(alu22+1)];
  var val34 = data3[(alu22+2)];
  var val35 = data3[(alu22+3)];
  var alu23 = (alu20+(gidx1*327680)+(lidx0*40960)+alu21);
  data0[alu23] = ((f16(acc0))+val32);
  data0[(alu23+1)] = ((f16(acc4))+val33);
  data0[(alu23+2)] = ((f16(acc8))+val34);
  data0[(alu23+3)] = ((f16(acc12))+val35);
  data0[(alu23+10240)] = ((f16(acc1))+val32);
  data0[(alu23+10241)] = ((f16(acc5))+val33);
  data0[(alu23+10242)] = ((f16(acc9))+val34);
  data0[(alu23+10243)] = ((f16(acc13))+val35);
  data0[(alu23+20480)] = ((f16(acc2))+val32);
  data0[(alu23+20481)] = ((f16(acc6))+val33);
  data0[(alu23+20482)] = ((f16(acc10))+val34);
  data0[(alu23+20483)] = ((f16(acc14))+val35);
  data0[(alu23+30720)] = ((f16(acc3))+val32);
  data0[(alu23+30721)] = ((f16(acc7))+val33);
  data0[(alu23+30722)] = ((f16(acc11))+val34);
  data0[(alu23+30723)] = ((f16(acc15))+val35);
}`;

const E_16_80_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 80 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = (alu0+(gidx1*81920)+(lidx0*10240)+alu1);
  var val0 = data1[alu2];
  var val1 = data1[(alu2+1)];
  var val2 = data1[(alu2+2)];
  var val3 = data1[(alu2+3)];
  var val4 = data1[(alu2+5120)];
  var val5 = data1[(alu2+5121)];
  var val6 = data1[(alu2+5122)];
  var val7 = data1[(alu2+5123)];
  var alu3 = (alu0+(gidx1*40960)+(lidx0*5120)+alu1);
  data0[alu3] = (val0*(1/(exp2(((val4+(val4*val4*val4*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val4);
  data0[(alu3+1)] = (val1*(1/(exp2(((val5+(val5*val5*val5*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val5);
  data0[(alu3+2)] = (val2*(1/(exp2(((val6+(val6*val6*val6*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val6);
  data0[(alu3+3)] = (val3*(1/(exp2(((val7+(val7*val7*val7*(f16(0.044715f))))*(f16(-2.302208198144325f))))+(f16(1.0f))))*val7);
}`;

const r_4_20_8_16_1280_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 4 */
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*327680)+(lidx1*20480)+alu0);
    var val0 = data3[alu1];
    var val1 = data3[(alu1+1)];
    var val2 = data3[(alu1+2)];
    var val3 = data3[(alu1+3)];
    var val4 = data3[(alu1+5120)];
    var val5 = data3[(alu1+5121)];
    var val6 = data3[(alu1+5122)];
    var val7 = data3[(alu1+5123)];
    var val8 = data3[(alu1+10240)];
    var val9 = data3[(alu1+10241)];
    var val10 = data3[(alu1+10242)];
    var val11 = data3[(alu1+10243)];
    var val12 = data3[(alu1+15360)];
    var val13 = data3[(alu1+15361)];
    var val14 = data3[(alu1+15362)];
    var val15 = data3[(alu1+15363)];
    var alu2 = ((gidx1*163840)+(lidx0*20480)+alu0);
    var val16 = data2[alu2];
    var val17 = data2[(alu2+1)];
    var val18 = data2[(alu2+2)];
    var val19 = data2[(alu2+3)];
    var val20 = data2[(alu2+5120)];
    var val21 = data2[(alu2+5121)];
    var val22 = data2[(alu2+5122)];
    var val23 = data2[(alu2+5123)];
    var val24 = data2[(alu2+10240)];
    var val25 = data2[(alu2+10241)];
    var val26 = data2[(alu2+10242)];
    var val27 = data2[(alu2+10243)];
    var val28 = data2[(alu2+15360)];
    var val29 = data2[(alu2+15361)];
    var val30 = data2[(alu2+15362)];
    var val31 = data2[(alu2+15363)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu20 = (gidx0<<6);
  var alu21 = (lidx1<<2);
  var alu22 = (alu20+(gidx1*40960)+(lidx0*5120)+alu21);
  var val32 = data1[alu22];
  var alu23 = (alu22+1);
  var val33 = data1[alu23];
  var alu24 = (alu22+2);
  var val34 = data1[alu24];
  var alu25 = (alu22+3);
  var val35 = data1[alu25];
  var alu26 = (alu22+1280);
  var val36 = data1[alu26];
  var alu27 = (alu22+1281);
  var val37 = data1[alu27];
  var alu28 = (alu22+1282);
  var val38 = data1[alu28];
  var alu29 = (alu22+1283);
  var val39 = data1[alu29];
  var alu30 = (alu22+2560);
  var val40 = data1[alu30];
  var alu31 = (alu22+2561);
  var val41 = data1[alu31];
  var alu32 = (alu22+2562);
  var val42 = data1[alu32];
  var alu33 = (alu22+2563);
  var val43 = data1[alu33];
  var alu34 = (alu22+3840);
  var val44 = data1[alu34];
  var alu35 = (alu22+3841);
  var val45 = data1[alu35];
  var alu36 = (alu22+3842);
  var val46 = data1[alu36];
  var alu37 = (alu22+3843);
  var val47 = data1[alu37];
  var alu38 = (alu20+alu21);
  var val48 = data4[alu38];
  var val49 = data4[(alu38+1)];
  var val50 = data4[(alu38+2)];
  var val51 = data4[(alu38+3)];
  data0[alu23] = (val33+(f16(acc4))+val49);
  data0[alu24] = (val34+(f16(acc8))+val50);
  data0[alu25] = (val35+(f16(acc12))+val51);
  data0[alu26] = (val36+(f16(acc1))+val48);
  data0[alu27] = (val37+(f16(acc5))+val49);
  data0[alu28] = (val38+(f16(acc9))+val50);
  data0[alu29] = (val39+(f16(acc13))+val51);
  data0[alu30] = (val40+(f16(acc2))+val48);
  data0[alu31] = (val41+(f16(acc6))+val49);
  data0[alu32] = (val42+(f16(acc10))+val50);
  data0[alu33] = (val43+(f16(acc14))+val51);
  data0[alu34] = (val44+(f16(acc3))+val48);
  data0[alu35] = (val45+(f16(acc7))+val49);
  data0[alu36] = (val46+(f16(acc11))+val50);
  data0[alu37] = (val47+(f16(acc15))+val51);
  data0[alu22] = (val32+(f16(acc0))+val48);
}`;

const r_2_40_8_16_320_4_4_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1*81920);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu1 = (ridx0<<2);
    var alu2 = ((gidx0*40960)+(lidx0*5120)+alu1);
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+1280)];
    var val5 = data2[(alu2+1281)];
    var val6 = data2[(alu2+1282)];
    var val7 = data2[(alu2+1283)];
    var val8 = data2[(alu2+2560)];
    var val9 = data2[(alu2+2561)];
    var val10 = data2[(alu2+2562)];
    var val11 = data2[(alu2+2563)];
    var val12 = data2[(alu2+3840)];
    var val13 = data2[(alu2+3841)];
    var val14 = data2[(alu2+3842)];
    var val15 = data2[(alu2+3843)];
    var alu3 = (alu0+(lidx1*5120)+alu1);
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+1280)];
    var val21 = data1[(alu3+1281)];
    var val22 = data1[(alu3+1282)];
    var val23 = data1[(alu3+1283)];
    var val24 = data1[(alu3+2560)];
    var val25 = data1[(alu3+2561)];
    var val26 = data1[(alu3+2562)];
    var val27 = data1[(alu3+2563)];
    var val28 = data1[(alu3+3840)];
    var val29 = data1[(alu3+3841)];
    var val30 = data1[(alu3+3842)];
    var val31 = data1[(alu3+3843)];
    acc0 = (acc0+(f32((val3*val19)))+(f32((val2*val18)))+(f32((val1*val17)))+(f32((val0*val16))));
    acc1 = (acc1+(f32((val3*val23)))+(f32((val2*val22)))+(f32((val1*val21)))+(f32((val0*val20))));
    acc2 = (acc2+(f32((val3*val27)))+(f32((val2*val26)))+(f32((val1*val25)))+(f32((val0*val24))));
    acc3 = (acc3+(f32((val3*val31)))+(f32((val2*val30)))+(f32((val1*val29)))+(f32((val0*val28))));
    acc4 = (acc4+(f32((val7*val19)))+(f32((val6*val18)))+(f32((val4*val16)))+(f32((val5*val17))));
    acc5 = (acc5+(f32((val7*val23)))+(f32((val6*val22)))+(f32((val4*val20)))+(f32((val5*val21))));
    acc6 = (acc6+(f32((val7*val27)))+(f32((val6*val26)))+(f32((val4*val24)))+(f32((val5*val25))));
    acc7 = (acc7+(f32((val7*val31)))+(f32((val6*val30)))+(f32((val4*val28)))+(f32((val5*val29))));
    acc8 = (acc8+(f32((val11*val19)))+(f32((val10*val18)))+(f32((val8*val16)))+(f32((val9*val17))));
    acc9 = (acc9+(f32((val11*val23)))+(f32((val10*val22)))+(f32((val8*val20)))+(f32((val9*val21))));
    acc10 = (acc10+(f32((val11*val27)))+(f32((val10*val26)))+(f32((val8*val24)))+(f32((val9*val25))));
    acc11 = (acc11+(f32((val11*val31)))+(f32((val10*val30)))+(f32((val8*val28)))+(f32((val9*val29))));
    acc12 = (acc12+(f32((val15*val19)))+(f32((val14*val18)))+(f32((val12*val16)))+(f32((val13*val17))));
    acc13 = (acc13+(f32((val15*val23)))+(f32((val14*val22)))+(f32((val12*val20)))+(f32((val13*val21))));
    acc14 = (acc14+(f32((val15*val27)))+(f32((val14*val26)))+(f32((val12*val24)))+(f32((val13*val25))));
    acc15 = (acc15+(f32((val15*val31)))+(f32((val14*val30)))+(f32((val12*val28)))+(f32((val13*val29))));
  }
  var alu21 = ((gidx0<<5)+(lidx0<<2));
  var val32 = data3[alu21];
  var val33 = data3[(alu21+1)];
  var val34 = data3[(alu21+2)];
  var val35 = data3[(alu21+3)];
  var alu22 = ((gidx0<<11)+alu0+(lidx0<<8)+(lidx1<<2));
  var val36 = data4[alu22];
  var alu23 = (alu22+1);
  var val37 = data4[alu23];
  var alu24 = (alu22+2);
  var val38 = data4[alu24];
  var alu25 = (alu22+3);
  var val39 = data4[alu25];
  var alu26 = (alu22+64);
  var val40 = data4[alu26];
  var alu27 = (alu22+65);
  var val41 = data4[alu27];
  var alu28 = (alu22+66);
  var val42 = data4[alu28];
  var alu29 = (alu22+67);
  var val43 = data4[alu29];
  var alu30 = (alu22+128);
  var val44 = data4[alu30];
  var alu31 = (alu22+129);
  var val45 = data4[alu31];
  var alu32 = (alu22+130);
  var val46 = data4[alu32];
  var alu33 = (alu22+131);
  var val47 = data4[alu33];
  var alu34 = (alu22+192);
  var val48 = data4[alu34];
  var alu35 = (alu22+193);
  var val49 = data4[alu35];
  var alu36 = (alu22+194);
  var val50 = data4[alu36];
  var alu37 = (alu22+195);
  var val51 = data4[alu37];
  data0[alu23] = (val37+(f16(acc1))+val32);
  data0[alu24] = (val38+(f16(acc2))+val32);
  data0[alu25] = (val39+(f16(acc3))+val32);
  data0[alu26] = (val40+(f16(acc4))+val33);
  data0[alu27] = (val41+(f16(acc5))+val33);
  data0[alu28] = (val42+(f16(acc6))+val33);
  data0[alu29] = (val43+(f16(acc7))+val33);
  data0[alu30] = (val44+(f16(acc8))+val34);
  data0[alu31] = (val45+(f16(acc9))+val34);
  data0[alu32] = (val46+(f16(acc10))+val34);
  data0[alu33] = (val47+(f16(acc11))+val34);
  data0[alu34] = (val48+(f16(acc12))+val35);
  data0[alu35] = (val49+(f16(acc13))+val35);
  data0[alu36] = (val50+(f16(acc14))+val35);
  data0[alu37] = (val51+(f16(acc15))+val35);
  data0[alu22] = (val36+(f16(acc0))+val32);
}`;

const E_2_320_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 320 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<160);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<9);
  var alu3 = (lidx0<<6);
  var alu4 = (lidx1<<2);
  var alu5 = (alu2+(gidx1*81920)+alu3+alu4);
  var val0 = select((f16(0.0f)), data1[alu5], alu0);
  var val1 = select((f16(0.0f)), data2[(alu5+-81920)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu5+-81919)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu5+-81918)], alu1);
  var val4 = select((f16(0.0f)), data2[(alu5+-81917)], alu1);
  var val5 = select((f16(0.0f)), data1[(alu5+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu5+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu5+3)], alu0);
  var alu6 = (alu2+(gidx1*163840)+alu3+alu4);
  data0[alu6] = (val0+val1);
  data0[(alu6+1)] = (val5+val2);
  data0[(alu6+2)] = (val6+val3);
  data0[(alu6+3)] = (val7+val4);
}`;

const E_2_320_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 320 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx0<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx1<<5)+(gidx0/10));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx0<<9)+(gidx1*163840)+(lidx0<<6)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_40_8_8_2_2560_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (lidx1<<3);
  var alu1 = (lidx2<<2);
  var alu2 = (lidx2<1);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (alu2!=true);
  var alu5 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 2560; ridx0++) {
    var alu6 = ((gidx0*737280)+(lidx0*92160)+(ridx0*9));
    var val0 = data2[alu6];
    var val1 = data2[(alu6+1)];
    var val2 = data2[(alu6+2)];
    var val3 = data2[(alu6+3)];
    var val4 = data2[(alu6+4)];
    var val5 = data2[(alu6+5)];
    var val6 = data2[(alu6+6)];
    var val7 = data2[(alu6+7)];
    var val8 = data2[(alu6+8)];
    var val9 = data2[(alu6+23040)];
    var val10 = data2[(alu6+23041)];
    var val11 = data2[(alu6+23042)];
    var val12 = data2[(alu6+23043)];
    var val13 = data2[(alu6+23044)];
    var val14 = data2[(alu6+23045)];
    var val15 = data2[(alu6+23046)];
    var val16 = data2[(alu6+23047)];
    var val17 = data2[(alu6+23048)];
    var val18 = data2[(alu6+46080)];
    var val19 = data2[(alu6+46081)];
    var val20 = data2[(alu6+46082)];
    var val21 = data2[(alu6+46083)];
    var val22 = data2[(alu6+46084)];
    var val23 = data2[(alu6+46085)];
    var val24 = data2[(alu6+46086)];
    var val25 = data2[(alu6+46087)];
    var val26 = data2[(alu6+46088)];
    var val27 = data2[(alu6+69120)];
    var val28 = data2[(alu6+69121)];
    var val29 = data2[(alu6+69122)];
    var val30 = data2[(alu6+69123)];
    var val31 = data2[(alu6+69124)];
    var val32 = data2[(alu6+69125)];
    var val33 = data2[(alu6+69126)];
    var val34 = data2[(alu6+69127)];
    var val35 = data2[(alu6+69128)];
    var alu7 = ((gidx1*163840)+(ridx0<<6)+alu0);
    var alu8 = (alu7+alu1);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-8)], alu3);
    var val38 = select((f16(0.0f)), data1[(alu8+-7)], alu3);
    var val39 = select((f16(0.0f)), data1[(alu8+-6)], alu3);
    var val40 = select((f16(0.0f)), data1[(alu8+-5)], alu3);
    var val41 = data1[(alu8+1)];
    var val42 = data1[(alu8+2)];
    var val43 = data1[(alu8+3)];
    var val44 = select((f16(0.0f)), data1[(alu8+8)], alu5);
    var val45 = select((f16(0.0f)), data1[(alu8+9)], alu5);
    var val46 = select((f16(0.0f)), data1[(alu8+10)], alu5);
    var val47 = select((f16(0.0f)), data1[(alu8+11)], alu5);
    var val48 = select((f16(0.0f)), data1[(alu7+-5)], (alu3&alu4));
    var val49 = select((f16(0.0f)), data1[(alu7+-4)], (alu2&alu3));
    var val50 = select((f16(0.0f)), data1[(alu7+3)], alu4);
    var val51 = select((f16(0.0f)), data1[(alu7+4)], alu2);
    var val52 = select((f16(0.0f)), data1[(alu7+11)], (alu5&alu4));
    var val53 = select((f16(0.0f)), data1[(alu7+12)], (alu5&alu2));
    acc0 = (acc0+(f32((val45*val8)))+(f32((val41*val5)))+(f32((val38*val2)))+(f32((val44*val7)))+(f32((val36*val4)))+(f32((val37*val1)))+(f32((val52*val6)))+(f32((val48*val0)))+(f32((val50*val3))));
    acc1 = (acc1+(f32((val45*val17)))+(f32((val41*val14)))+(f32((val38*val11)))+(f32((val44*val16)))+(f32((val36*val13)))+(f32((val37*val10)))+(f32((val52*val15)))+(f32((val48*val9)))+(f32((val50*val12))));
    acc2 = (acc2+(f32((val45*val26)))+(f32((val41*val23)))+(f32((val38*val20)))+(f32((val44*val25)))+(f32((val36*val22)))+(f32((val37*val19)))+(f32((val52*val24)))+(f32((val48*val18)))+(f32((val50*val21))));
    acc3 = (acc3+(f32((val45*val35)))+(f32((val41*val32)))+(f32((val38*val29)))+(f32((val44*val34)))+(f32((val36*val31)))+(f32((val37*val28)))+(f32((val52*val33)))+(f32((val48*val27)))+(f32((val50*val30))));
    acc4 = (acc4+(f32((val46*val8)))+(f32((val42*val5)))+(f32((val39*val2)))+(f32((val45*val7)))+(f32((val41*val4)))+(f32((val38*val1)))+(f32((val44*val6)))+(f32((val37*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val46*val17)))+(f32((val42*val14)))+(f32((val39*val11)))+(f32((val45*val16)))+(f32((val41*val13)))+(f32((val38*val10)))+(f32((val44*val15)))+(f32((val37*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val46*val26)))+(f32((val42*val23)))+(f32((val39*val20)))+(f32((val45*val25)))+(f32((val41*val22)))+(f32((val38*val19)))+(f32((val44*val24)))+(f32((val37*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val46*val35)))+(f32((val42*val32)))+(f32((val39*val29)))+(f32((val45*val34)))+(f32((val41*val31)))+(f32((val38*val28)))+(f32((val44*val33)))+(f32((val37*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val47*val8)))+(f32((val43*val5)))+(f32((val40*val2)))+(f32((val46*val7)))+(f32((val42*val4)))+(f32((val39*val1)))+(f32((val45*val6)))+(f32((val38*val0)))+(f32((val41*val3))));
    acc9 = (acc9+(f32((val47*val17)))+(f32((val43*val14)))+(f32((val40*val11)))+(f32((val46*val16)))+(f32((val42*val13)))+(f32((val39*val10)))+(f32((val45*val15)))+(f32((val38*val9)))+(f32((val41*val12))));
    acc10 = (acc10+(f32((val47*val26)))+(f32((val43*val23)))+(f32((val40*val20)))+(f32((val46*val25)))+(f32((val42*val22)))+(f32((val39*val19)))+(f32((val45*val24)))+(f32((val38*val18)))+(f32((val41*val21))));
    acc11 = (acc11+(f32((val47*val35)))+(f32((val43*val32)))+(f32((val40*val29)))+(f32((val46*val34)))+(f32((val42*val31)))+(f32((val39*val28)))+(f32((val45*val33)))+(f32((val38*val27)))+(f32((val41*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val51*val5)))+(f32((val49*val2)))+(f32((val47*val7)))+(f32((val43*val4)))+(f32((val40*val1)))+(f32((val46*val6)))+(f32((val39*val0)))+(f32((val42*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val51*val14)))+(f32((val49*val11)))+(f32((val47*val16)))+(f32((val43*val13)))+(f32((val40*val10)))+(f32((val46*val15)))+(f32((val39*val9)))+(f32((val42*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val51*val23)))+(f32((val49*val20)))+(f32((val47*val25)))+(f32((val43*val22)))+(f32((val40*val19)))+(f32((val46*val24)))+(f32((val39*val18)))+(f32((val42*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val51*val32)))+(f32((val49*val29)))+(f32((val47*val34)))+(f32((val43*val31)))+(f32((val40*val28)))+(f32((val46*val33)))+(f32((val39*val27)))+(f32((val42*val30))));
  }
  var alu26 = ((gidx0<<5)+(lidx0<<2));
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx0<<11)+(gidx1*81920)+(lidx0<<8)+alu0+alu1);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+64)] = (val57+(f16(acc1))+val56);
  data0[(alu30+65)] = (val57+(f16(acc5))+val56);
  data0[(alu30+66)] = (val57+(f16(acc9))+val56);
  data0[(alu30+67)] = (val57+(f16(acc13))+val56);
  data0[(alu30+128)] = (val59+(f16(acc2))+val58);
  data0[(alu30+129)] = (val59+(f16(acc6))+val58);
  data0[(alu30+130)] = (val59+(f16(acc10))+val58);
  data0[(alu30+131)] = (val59+(f16(acc14))+val58);
  data0[(alu30+192)] = (val61+(f16(acc3))+val60);
  data0[(alu30+193)] = (val61+(f16(acc7))+val60);
  data0[(alu30+194)] = (val61+(f16(acc11))+val60);
  data0[(alu30+195)] = (val61+(f16(acc15))+val60);
}`;

const r_2_40_8_8_2_1280_4_4_3_3n3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (gidx1*81920);
  var alu1 = (lidx1<<3);
  var alu2 = (lidx2<<2);
  var alu3 = (lidx2<1);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (alu3!=true);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx0*368640)+(lidx0*46080)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+11520)];
    var val10 = data2[(alu7+11521)];
    var val11 = data2[(alu7+11522)];
    var val12 = data2[(alu7+11523)];
    var val13 = data2[(alu7+11524)];
    var val14 = data2[(alu7+11525)];
    var val15 = data2[(alu7+11526)];
    var val16 = data2[(alu7+11527)];
    var val17 = data2[(alu7+11528)];
    var val18 = data2[(alu7+23040)];
    var val19 = data2[(alu7+23041)];
    var val20 = data2[(alu7+23042)];
    var val21 = data2[(alu7+23043)];
    var val22 = data2[(alu7+23044)];
    var val23 = data2[(alu7+23045)];
    var val24 = data2[(alu7+23046)];
    var val25 = data2[(alu7+23047)];
    var val26 = data2[(alu7+23048)];
    var val27 = data2[(alu7+34560)];
    var val28 = data2[(alu7+34561)];
    var val29 = data2[(alu7+34562)];
    var val30 = data2[(alu7+34563)];
    var val31 = data2[(alu7+34564)];
    var val32 = data2[(alu7+34565)];
    var val33 = data2[(alu7+34566)];
    var val34 = data2[(alu7+34567)];
    var val35 = data2[(alu7+34568)];
    var alu8 = (alu0+(ridx0<<6)+alu1);
    var alu9 = (alu8+alu2);
    var val36 = data1[alu9];
    var val37 = select((f16(0.0f)), data1[(alu9+-8)], alu4);
    var val38 = select((f16(0.0f)), data1[(alu9+-7)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu9+-6)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu9+-5)], alu4);
    var val41 = data1[(alu9+1)];
    var val42 = data1[(alu9+2)];
    var val43 = data1[(alu9+3)];
    var val44 = select((f16(0.0f)), data1[(alu9+8)], alu6);
    var val45 = select((f16(0.0f)), data1[(alu9+9)], alu6);
    var val46 = select((f16(0.0f)), data1[(alu9+10)], alu6);
    var val47 = select((f16(0.0f)), data1[(alu9+11)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+-5)], (alu4&alu5));
    var val49 = select((f16(0.0f)), data1[(alu8+-4)], (alu3&alu4));
    var val50 = select((f16(0.0f)), data1[(alu8+3)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+4)], alu3);
    var val52 = select((f16(0.0f)), data1[(alu8+11)], (alu6&alu5));
    var val53 = select((f16(0.0f)), data1[(alu8+12)], (alu6&alu3));
    acc0 = (acc0+(f32((val8*val45)))+(f32((val5*val41)))+(f32((val2*val38)))+(f32((val7*val44)))+(f32((val4*val36)))+(f32((val1*val37)))+(f32((val6*val52)))+(f32((val3*val50)))+(f32((val0*val48))));
    acc1 = (acc1+(f32((val8*val46)))+(f32((val5*val42)))+(f32((val2*val39)))+(f32((val7*val45)))+(f32((val4*val41)))+(f32((val1*val38)))+(f32((val6*val44)))+(f32((val3*val36)))+(f32((val0*val37))));
    acc2 = (acc2+(f32((val8*val47)))+(f32((val5*val43)))+(f32((val2*val40)))+(f32((val7*val46)))+(f32((val4*val42)))+(f32((val1*val39)))+(f32((val6*val45)))+(f32((val3*val41)))+(f32((val0*val38))));
    acc3 = (acc3+(f32((val8*val53)))+(f32((val5*val51)))+(f32((val2*val49)))+(f32((val7*val47)))+(f32((val4*val43)))+(f32((val1*val40)))+(f32((val6*val46)))+(f32((val3*val42)))+(f32((val0*val39))));
    acc4 = (acc4+(f32((val17*val45)))+(f32((val14*val41)))+(f32((val11*val38)))+(f32((val16*val44)))+(f32((val13*val36)))+(f32((val10*val37)))+(f32((val15*val52)))+(f32((val9*val48)))+(f32((val12*val50))));
    acc5 = (acc5+(f32((val17*val46)))+(f32((val14*val42)))+(f32((val11*val39)))+(f32((val16*val45)))+(f32((val13*val41)))+(f32((val10*val38)))+(f32((val15*val44)))+(f32((val9*val37)))+(f32((val12*val36))));
    acc6 = (acc6+(f32((val17*val47)))+(f32((val14*val43)))+(f32((val11*val40)))+(f32((val16*val46)))+(f32((val13*val42)))+(f32((val10*val39)))+(f32((val15*val45)))+(f32((val9*val38)))+(f32((val12*val41))));
    acc7 = (acc7+(f32((val17*val53)))+(f32((val14*val51)))+(f32((val11*val49)))+(f32((val16*val47)))+(f32((val13*val43)))+(f32((val10*val40)))+(f32((val15*val46)))+(f32((val9*val39)))+(f32((val12*val42))));
    acc8 = (acc8+(f32((val26*val45)))+(f32((val23*val41)))+(f32((val20*val38)))+(f32((val25*val44)))+(f32((val22*val36)))+(f32((val19*val37)))+(f32((val24*val52)))+(f32((val18*val48)))+(f32((val21*val50))));
    acc9 = (acc9+(f32((val26*val46)))+(f32((val23*val42)))+(f32((val20*val39)))+(f32((val25*val45)))+(f32((val22*val41)))+(f32((val19*val38)))+(f32((val24*val44)))+(f32((val18*val37)))+(f32((val21*val36))));
    acc10 = (acc10+(f32((val26*val47)))+(f32((val23*val43)))+(f32((val20*val40)))+(f32((val25*val46)))+(f32((val22*val42)))+(f32((val19*val39)))+(f32((val24*val45)))+(f32((val18*val38)))+(f32((val21*val41))));
    acc11 = (acc11+(f32((val26*val53)))+(f32((val23*val51)))+(f32((val20*val49)))+(f32((val25*val47)))+(f32((val22*val43)))+(f32((val19*val40)))+(f32((val24*val46)))+(f32((val18*val39)))+(f32((val21*val42))));
    acc12 = (acc12+(f32((val35*val45)))+(f32((val32*val41)))+(f32((val29*val38)))+(f32((val34*val44)))+(f32((val31*val36)))+(f32((val28*val37)))+(f32((val33*val52)))+(f32((val27*val48)))+(f32((val30*val50))));
    acc13 = (acc13+(f32((val35*val46)))+(f32((val32*val42)))+(f32((val29*val39)))+(f32((val34*val45)))+(f32((val31*val41)))+(f32((val28*val38)))+(f32((val33*val44)))+(f32((val27*val37)))+(f32((val30*val36))));
    acc14 = (acc14+(f32((val35*val47)))+(f32((val32*val43)))+(f32((val29*val40)))+(f32((val34*val46)))+(f32((val31*val42)))+(f32((val28*val39)))+(f32((val33*val45)))+(f32((val27*val38)))+(f32((val30*val41))));
    acc15 = (acc15+(f32((val35*val53)))+(f32((val32*val51)))+(f32((val29*val49)))+(f32((val34*val47)))+(f32((val31*val43)))+(f32((val28*val40)))+(f32((val33*val46)))+(f32((val27*val39)))+(f32((val30*val42))));
  }
  var alu27 = ((gidx0<<11)+alu0+(lidx0<<8)+alu1+alu2);
  data0[alu27] = acc0;
  data0[(alu27+1)] = acc1;
  data0[(alu27+2)] = acc2;
  data0[(alu27+3)] = acc3;
  data0[(alu27+64)] = acc4;
  data0[(alu27+65)] = acc5;
  data0[(alu27+66)] = acc6;
  data0[(alu27+67)] = acc7;
  data0[(alu27+128)] = acc8;
  data0[(alu27+129)] = acc9;
  data0[(alu27+130)] = acc10;
  data0[(alu27+131)] = acc11;
  data0[(alu27+192)] = acc12;
  data0[(alu27+193)] = acc13;
  data0[(alu27+194)] = acc14;
  data0[(alu27+195)] = acc15;
}`;

const r_2_40_8_16_640_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 40 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx1<<2);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu1 = ((gidx0*81920)+(lidx0*10240)+(ridx0<<2));
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+2560)];
    var val5 = data2[(alu1+2561)];
    var val6 = data2[(alu1+2562)];
    var val7 = data2[(alu1+2563)];
    var val8 = data2[(alu1+5120)];
    var val9 = data2[(alu1+5121)];
    var val10 = data2[(alu1+5122)];
    var val11 = data2[(alu1+5123)];
    var val12 = data2[(alu1+7680)];
    var val13 = data2[(alu1+7681)];
    var val14 = data2[(alu1+7682)];
    var val15 = data2[(alu1+7683)];
    var alu2 = ((gidx1*163840)+alu0+(ridx0<<8));
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+64)];
    var val21 = data1[(alu2+65)];
    var val22 = data1[(alu2+66)];
    var val23 = data1[(alu2+67)];
    var val24 = data1[(alu2+128)];
    var val25 = data1[(alu2+129)];
    var val26 = data1[(alu2+130)];
    var val27 = data1[(alu2+131)];
    var val28 = data1[(alu2+192)];
    var val29 = data1[(alu2+193)];
    var val30 = data1[(alu2+194)];
    var val31 = data1[(alu2+195)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
  }
  var alu20 = ((gidx0<<5)+(lidx0<<2));
  var val32 = data3[alu20];
  var val33 = data5[alu20];
  var alu21 = (alu20+1);
  var val34 = data3[alu21];
  var val35 = data5[alu21];
  var alu22 = (alu20+2);
  var val36 = data3[alu22];
  var val37 = data5[alu22];
  var alu23 = (alu20+3);
  var val38 = data3[alu23];
  var val39 = data5[alu23];
  var alu24 = ((gidx0<<11)+(gidx1*81920)+(lidx0<<8)+alu0);
  var val40 = data4[alu24];
  var alu25 = (alu24+1);
  var val41 = data4[alu25];
  var alu26 = (alu24+2);
  var val42 = data4[alu26];
  var alu27 = (alu24+3);
  var val43 = data4[alu27];
  var alu28 = (alu24+64);
  var val44 = data4[alu28];
  var alu29 = (alu24+65);
  var val45 = data4[alu29];
  var alu30 = (alu24+66);
  var val46 = data4[alu30];
  var alu31 = (alu24+67);
  var val47 = data4[alu31];
  var alu32 = (alu24+128);
  var val48 = data4[alu32];
  var alu33 = (alu24+129);
  var val49 = data4[alu33];
  var alu34 = (alu24+130);
  var val50 = data4[alu34];
  var alu35 = (alu24+131);
  var val51 = data4[alu35];
  var alu36 = (alu24+192);
  var val52 = data4[alu36];
  var alu37 = (alu24+193);
  var val53 = data4[alu37];
  var alu38 = (alu24+194);
  var val54 = data4[alu38];
  var alu39 = (alu24+195);
  var val55 = data4[alu39];
  data0[alu25] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu26] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu27] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu28] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu29] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu30] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu31] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu32] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu33] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu34] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu35] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu36] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu37] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu38] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu39] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu24] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const r_2_160_2_16_4_1280_4_4_3_3n3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = ((lidx1<1)!=true);
  var alu1 = ((lidx2<1)!=true);
  var alu2 = (lidx1<15);
  var alu3 = (lidx2<3);
  var alu4 = (lidx2<<1);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu5 = ((gidx0*92160)+(lidx0*46080)+(ridx0*9));
    var val0 = data2[alu5];
    var val1 = data2[(alu5+1)];
    var val2 = data2[(alu5+2)];
    var val3 = data2[(alu5+3)];
    var val4 = data2[(alu5+4)];
    var val5 = data2[(alu5+5)];
    var val6 = data2[(alu5+6)];
    var val7 = data2[(alu5+7)];
    var val8 = data2[(alu5+8)];
    var val9 = data2[(alu5+11520)];
    var val10 = data2[(alu5+11521)];
    var val11 = data2[(alu5+11522)];
    var val12 = data2[(alu5+11523)];
    var val13 = data2[(alu5+11524)];
    var val14 = data2[(alu5+11525)];
    var val15 = data2[(alu5+11526)];
    var val16 = data2[(alu5+11527)];
    var val17 = data2[(alu5+11528)];
    var val18 = data2[(alu5+23040)];
    var val19 = data2[(alu5+23041)];
    var val20 = data2[(alu5+23042)];
    var val21 = data2[(alu5+23043)];
    var val22 = data2[(alu5+23044)];
    var val23 = data2[(alu5+23045)];
    var val24 = data2[(alu5+23046)];
    var val25 = data2[(alu5+23047)];
    var val26 = data2[(alu5+23048)];
    var val27 = data2[(alu5+34560)];
    var val28 = data2[(alu5+34561)];
    var val29 = data2[(alu5+34562)];
    var val30 = data2[(alu5+34563)];
    var val31 = data2[(alu5+34564)];
    var val32 = data2[(alu5+34565)];
    var val33 = data2[(alu5+34566)];
    var val34 = data2[(alu5+34567)];
    var val35 = data2[(alu5+34568)];
    var alu6 = ((gidx1*81920)+(ridx0<<6));
    var alu7 = (alu6+((lidx1>>1)<<3)+alu4);
    var val36 = data1[alu7];
    var val37 = select((f16(0.0f)), data1[(alu7+-1)], alu1);
    var val38 = data1[(alu7+1)];
    var val39 = select((f16(0.0f)), data1[(alu7+2)], alu3);
    var alu8 = (alu6+(((lidx1+1)>>1)<<3)+alu4);
    var val40 = select((f16(0.0f)), data1[alu8], alu2);
    var val41 = select((f16(0.0f)), data1[(alu8+-9)], (alu0&alu1));
    var val42 = select((f16(0.0f)), data1[(alu8+-8)], alu0);
    var val43 = select((f16(0.0f)), data1[(alu8+-7)], alu0);
    var val44 = select((f16(0.0f)), data1[(alu8+-6)], (alu3&alu0));
    var val45 = select((f16(0.0f)), data1[(alu8+-1)], (alu2&alu1));
    var val46 = select((f16(0.0f)), data1[(alu8+1)], alu2);
    var val47 = select((f16(0.0f)), data1[(alu8+2)], (alu2&alu3));
    var cast0 = (f32((val38*val4)));
    var cast1 = (f32((val38*val5)));
    var cast2 = (f32((val38*val13)));
    var cast3 = (f32((val38*val14)));
    var cast4 = (f32((val38*val22)));
    var cast5 = (f32((val38*val23)));
    var cast6 = (f32((val38*val31)));
    var cast7 = (f32((val38*val32)));
    var cast8 = (f32((val42*val1)));
    var cast9 = (f32((val42*val10)));
    var cast10 = (f32((val42*val19)));
    var cast11 = (f32((val42*val28)));
    var cast12 = (f32((val43*val1)));
    var cast13 = (f32((val43*val2)));
    var cast14 = (f32((val43*val10)));
    var cast15 = (f32((val43*val11)));
    var cast16 = (f32((val43*val19)));
    var cast17 = (f32((val43*val20)));
    var cast18 = (f32((val43*val28)));
    var cast19 = (f32((val43*val29)));
    var cast20 = (f32((val46*val7)));
    var cast21 = (f32((val46*val8)));
    var cast22 = (f32((val46*val16)));
    var cast23 = (f32((val46*val17)));
    var cast24 = (f32((val46*val25)));
    var cast25 = (f32((val46*val26)));
    var cast26 = (f32((val46*val34)));
    var cast27 = (f32((val46*val35)));
    var cast28 = (f32((val36*val4)));
    var cast29 = (f32((val36*val13)));
    var cast30 = (f32((val36*val22)));
    var cast31 = (f32((val36*val31)));
    var alu9 = ((f32((val40*val6)))+(f32((val42*val0)))+(f32((val36*val3))));
    var cast32 = (f32((val40*val7)));
    var alu10 = ((f32((val40*val15)))+(f32((val42*val9)))+(f32((val36*val12))));
    var cast33 = (f32((val40*val16)));
    var alu11 = ((f32((val40*val24)))+(f32((val42*val18)))+(f32((val36*val21))));
    var cast34 = (f32((val40*val25)));
    var alu12 = ((f32((val40*val33)))+(f32((val42*val27)))+(f32((val36*val30))));
    var cast35 = (f32((val40*val34)));
    acc0 = (acc0+(f32((val40*val8)))+(f32((val36*val5)))+(f32((val42*val2)))+cast32+cast28+cast8+(f32((val45*val6)))+(f32((val37*val3)))+(f32((val41*val0))));
    acc1 = (acc1+(f32((val40*val17)))+(f32((val36*val14)))+(f32((val42*val11)))+cast33+cast29+cast9+(f32((val45*val15)))+(f32((val37*val12)))+(f32((val41*val9))));
    acc2 = (acc2+(f32((val40*val26)))+(f32((val36*val23)))+(f32((val42*val20)))+cast34+cast30+cast10+(f32((val45*val24)))+(f32((val37*val21)))+(f32((val41*val18))));
    acc3 = (acc3+(f32((val40*val35)))+(f32((val36*val32)))+(f32((val42*val29)))+cast35+cast31+cast11+(f32((val45*val33)))+(f32((val37*val30)))+(f32((val41*val27))));
    acc4 = (acc4+cast21+cast1+cast13+cast32+cast28+cast8+alu9);
    acc5 = (acc5+cast23+cast3+cast15+cast33+cast29+cast9+alu10);
    acc6 = (acc6+cast25+cast5+cast17+cast34+cast30+cast10+alu11);
    acc7 = (acc7+cast27+cast7+cast19+cast35+cast31+cast11+alu12);
    acc8 = (acc8+cast21+cast1+cast13+cast20+cast0+cast12+alu9);
    acc9 = (acc9+cast23+cast3+cast15+cast22+cast2+cast14+alu10);
    acc10 = (acc10+cast25+cast5+cast17+cast24+cast4+cast16+alu11);
    acc11 = (acc11+cast27+cast7+cast19+cast26+cast6+cast18+alu12);
    acc12 = (acc12+(f32((val47*val8)))+(f32((val39*val5)))+(f32((val44*val2)))+cast20+cast0+cast12+(f32((val46*val6)))+(f32((val38*val3)))+(f32((val43*val0))));
    acc13 = (acc13+(f32((val47*val17)))+(f32((val39*val14)))+(f32((val44*val11)))+cast22+cast2+cast14+(f32((val46*val15)))+(f32((val38*val12)))+(f32((val43*val9))));
    acc14 = (acc14+(f32((val47*val26)))+(f32((val39*val23)))+(f32((val44*val20)))+cast24+cast4+cast16+(f32((val46*val24)))+(f32((val38*val21)))+(f32((val43*val18))));
    acc15 = (acc15+(f32((val47*val35)))+(f32((val39*val32)))+(f32((val44*val29)))+cast26+cast6+cast18+(f32((val46*val33)))+(f32((val38*val30)))+(f32((val43*val27))));
  }
  var alu30 = ((gidx0<<3)+(lidx0<<2));
  var val48 = data3[alu30];
  var val49 = data3[(alu30+1)];
  var val50 = data3[(alu30+2)];
  var val51 = data3[(alu30+3)];
  var alu31 = ((gidx0<<11)+(gidx1*327680)+(lidx0<<10)+(lidx1<<4)+(lidx2<<2));
  data0[alu31] = ((f16(acc0))+val48);
  data0[(alu31+1)] = ((f16(acc4))+val48);
  data0[(alu31+2)] = ((f16(acc8))+val48);
  data0[(alu31+3)] = ((f16(acc12))+val48);
  data0[(alu31+256)] = ((f16(acc1))+val49);
  data0[(alu31+257)] = ((f16(acc5))+val49);
  data0[(alu31+258)] = ((f16(acc9))+val49);
  data0[(alu31+259)] = ((f16(acc13))+val49);
  data0[(alu31+512)] = ((f16(acc2))+val50);
  data0[(alu31+513)] = ((f16(acc6))+val50);
  data0[(alu31+514)] = ((f16(acc10))+val50);
  data0[(alu31+515)] = ((f16(acc14))+val50);
  data0[(alu31+768)] = ((f16(acc3))+val51);
  data0[(alu31+769)] = ((f16(acc7))+val51);
  data0[(alu31+770)] = ((f16(acc11))+val51);
  data0[(alu31+771)] = ((f16(acc15))+val51);
}`;

const E_2_320_4_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 320 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<160);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<11);
  var alu4 = (lidx0<<8);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*327680)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data1[alu6], alu0);
  var val1 = select((f16(0.0f)), data2[(alu6+-327680)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-327679)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-327678)], alu1);
  var val4 = select((f16(0.0f)), data2[(alu6+-327677)], alu1);
  var val5 = select((f16(0.0f)), data1[(alu6+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu6+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu6+3)], alu0);
  var alu7 = (alu3+(gidx2*655360)+alu2+alu4+alu5);
  data0[alu7] = (val0+val1);
  data0[(alu7+1)] = (val5+val2);
  data0[(alu7+2)] = (val6+val3);
  data0[(alu7+3)] = (val7+val4);
}`;

const E_2_320_4_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 320 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(gidx1/10));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<11)+(gidx2*655360)+(gidx0<<6)+(lidx0<<8)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_4_2560_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (lidx1<<4);
  var alu1 = (lidx2<<2);
  var alu2 = ((lidx1<1)!=true);
  var alu3 = ((lidx2<1)!=true);
  var alu4 = (lidx1<15);
  var alu5 = (lidx2<3);
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
  for (var ridx0 = 0; ridx0 < 2560; ridx0++) {
    var alu6 = ((gidx0*184320)+(lidx0*92160)+(ridx0*9));
    var val0 = data2[alu6];
    var val1 = data2[(alu6+1)];
    var val2 = data2[(alu6+2)];
    var val3 = data2[(alu6+3)];
    var val4 = data2[(alu6+4)];
    var val5 = data2[(alu6+5)];
    var val6 = data2[(alu6+6)];
    var val7 = data2[(alu6+7)];
    var val8 = data2[(alu6+8)];
    var val9 = data2[(alu6+23040)];
    var val10 = data2[(alu6+23041)];
    var val11 = data2[(alu6+23042)];
    var val12 = data2[(alu6+23043)];
    var val13 = data2[(alu6+23044)];
    var val14 = data2[(alu6+23045)];
    var val15 = data2[(alu6+23046)];
    var val16 = data2[(alu6+23047)];
    var val17 = data2[(alu6+23048)];
    var val18 = data2[(alu6+46080)];
    var val19 = data2[(alu6+46081)];
    var val20 = data2[(alu6+46082)];
    var val21 = data2[(alu6+46083)];
    var val22 = data2[(alu6+46084)];
    var val23 = data2[(alu6+46085)];
    var val24 = data2[(alu6+46086)];
    var val25 = data2[(alu6+46087)];
    var val26 = data2[(alu6+46088)];
    var val27 = data2[(alu6+69120)];
    var val28 = data2[(alu6+69121)];
    var val29 = data2[(alu6+69122)];
    var val30 = data2[(alu6+69123)];
    var val31 = data2[(alu6+69124)];
    var val32 = data2[(alu6+69125)];
    var val33 = data2[(alu6+69126)];
    var val34 = data2[(alu6+69127)];
    var val35 = data2[(alu6+69128)];
    var alu7 = ((gidx1*655360)+(ridx0<<8)+alu0+alu1);
    var val36 = data1[alu7];
    var val37 = select((f16(0.0f)), data1[(alu7+-17)], (alu2&alu3));
    var val38 = select((f16(0.0f)), data1[(alu7+-16)], alu2);
    var val39 = select((f16(0.0f)), data1[(alu7+-15)], alu2);
    var val40 = select((f16(0.0f)), data1[(alu7+-14)], alu2);
    var val41 = select((f16(0.0f)), data1[(alu7+-13)], alu2);
    var val42 = select((f16(0.0f)), data1[(alu7+-12)], (alu5&alu2));
    var val43 = select((f16(0.0f)), data1[(alu7+-1)], alu3);
    var val44 = data1[(alu7+1)];
    var val45 = data1[(alu7+2)];
    var val46 = data1[(alu7+3)];
    var val47 = select((f16(0.0f)), data1[(alu7+4)], alu5);
    var val48 = select((f16(0.0f)), data1[(alu7+15)], (alu4&alu3));
    var val49 = select((f16(0.0f)), data1[(alu7+16)], alu4);
    var val50 = select((f16(0.0f)), data1[(alu7+17)], alu4);
    var val51 = select((f16(0.0f)), data1[(alu7+18)], alu4);
    var val52 = select((f16(0.0f)), data1[(alu7+19)], alu4);
    var val53 = select((f16(0.0f)), data1[(alu7+20)], (alu4&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu25 = ((gidx0<<3)+(lidx0<<2));
  var val54 = data3[alu25];
  var val55 = data4[alu25];
  var alu26 = (alu25+1);
  var val56 = data3[alu26];
  var val57 = data4[alu26];
  var alu27 = (alu25+2);
  var val58 = data3[alu27];
  var val59 = data4[alu27];
  var alu28 = (alu25+3);
  var val60 = data3[alu28];
  var val61 = data4[alu28];
  var alu29 = ((gidx0<<11)+(gidx1*327680)+(lidx0<<10)+alu0+alu1);
  data0[alu29] = (val55+(f16(acc0))+val54);
  data0[(alu29+1)] = (val55+(f16(acc4))+val54);
  data0[(alu29+2)] = (val55+(f16(acc8))+val54);
  data0[(alu29+3)] = (val55+(f16(acc12))+val54);
  data0[(alu29+256)] = (val57+(f16(acc1))+val56);
  data0[(alu29+257)] = (val57+(f16(acc5))+val56);
  data0[(alu29+258)] = (val57+(f16(acc9))+val56);
  data0[(alu29+259)] = (val57+(f16(acc13))+val56);
  data0[(alu29+512)] = (val59+(f16(acc2))+val58);
  data0[(alu29+513)] = (val59+(f16(acc6))+val58);
  data0[(alu29+514)] = (val59+(f16(acc10))+val58);
  data0[(alu29+515)] = (val59+(f16(acc14))+val58);
  data0[(alu29+768)] = (val61+(f16(acc3))+val60);
  data0[(alu29+769)] = (val61+(f16(acc7))+val60);
  data0[(alu29+770)] = (val61+(f16(acc11))+val60);
  data0[(alu29+771)] = (val61+(f16(acc15))+val60);
}`;

const r_2_40_4_8_16_640_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu2 = ((gidx1*81920)+(lidx0*10240)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+2560)];
    var val5 = data2[(alu2+2561)];
    var val6 = data2[(alu2+2562)];
    var val7 = data2[(alu2+2563)];
    var val8 = data2[(alu2+5120)];
    var val9 = data2[(alu2+5121)];
    var val10 = data2[(alu2+5122)];
    var val11 = data2[(alu2+5123)];
    var val12 = data2[(alu2+7680)];
    var val13 = data2[(alu2+7681)];
    var val14 = data2[(alu2+7682)];
    var val15 = data2[(alu2+7683)];
    var alu3 = (alu0+(gidx2*655360)+alu1+(ridx0<<10));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+256)];
    var val21 = data1[(alu3+257)];
    var val22 = data1[(alu3+258)];
    var val23 = data1[(alu3+259)];
    var val24 = data1[(alu3+512)];
    var val25 = data1[(alu3+513)];
    var val26 = data1[(alu3+514)];
    var val27 = data1[(alu3+515)];
    var val28 = data1[(alu3+768)];
    var val29 = data1[(alu3+769)];
    var val30 = data1[(alu3+770)];
    var val31 = data1[(alu3+771)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<13)+(gidx2*327680)+alu0+(lidx0<<10)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+256);
  var val44 = data4[alu29];
  var alu30 = (alu25+257);
  var val45 = data4[alu30];
  var alu31 = (alu25+258);
  var val46 = data4[alu31];
  var alu32 = (alu25+259);
  var val47 = data4[alu32];
  var alu33 = (alu25+512);
  var val48 = data4[alu33];
  var alu34 = (alu25+513);
  var val49 = data4[alu34];
  var alu35 = (alu25+514);
  var val50 = data4[alu35];
  var alu36 = (alu25+515);
  var val51 = data4[alu36];
  var alu37 = (alu25+768);
  var val52 = data4[alu37];
  var alu38 = (alu25+769);
  var val53 = data4[alu38];
  var alu39 = (alu25+770);
  var val54 = data4[alu39];
  var alu40 = (alu25+771);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const E_2_240_4_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 240 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<160);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<11);
  var alu4 = (lidx0<<8);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*163840)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data2[(alu6+-327680)], alu1);
  var val1 = select((f16(0.0f)), data2[(alu6+-327679)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-327678)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-327677)], alu1);
  var alu7 = (alu3+(gidx2*327680)+alu2+alu4+alu5);
  var val4 = select((f16(0.0f)), data1[alu7], alu0);
  var val5 = select((f16(0.0f)), data1[(alu7+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu7+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu7+3)], alu0);
  var alu8 = (alu3+(gidx2*491520)+alu2+alu4+alu5);
  data0[alu8] = (val4+val0);
  data0[(alu8+1)] = (val5+val1);
  data0[(alu8+2)] = (val6+val2);
  data0[(alu8+3)] = (val7+val3);
}`;

const r_64_16_960 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 960; ridx0++) {
    var val0 = data1[((gidx0*15360)+(lidx0*960)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*6.510417006211355e-05f)));
  }
}`;

const r_64_16_960n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 960; ridx0++) {
    var val1 = data1[((gidx0*15360)+(lidx0*960)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*6.510417006211355e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_240_4_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 240 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/60));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<11)+(gidx2*491520)+(gidx0<<6)+(lidx0<<8)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_4_1920_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 160 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var alu0 = (lidx1<<4);
  var alu1 = (lidx2<<2);
  var alu2 = ((lidx1<1)!=true);
  var alu3 = ((lidx2<1)!=true);
  var alu4 = (lidx1<15);
  var alu5 = (lidx2<3);
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
  for (var ridx0 = 0; ridx0 < 1920; ridx0++) {
    var alu6 = ((gidx0*138240)+(lidx0*69120)+(ridx0*9));
    var val0 = data2[alu6];
    var val1 = data2[(alu6+1)];
    var val2 = data2[(alu6+2)];
    var val3 = data2[(alu6+3)];
    var val4 = data2[(alu6+4)];
    var val5 = data2[(alu6+5)];
    var val6 = data2[(alu6+6)];
    var val7 = data2[(alu6+7)];
    var val8 = data2[(alu6+8)];
    var val9 = data2[(alu6+17280)];
    var val10 = data2[(alu6+17281)];
    var val11 = data2[(alu6+17282)];
    var val12 = data2[(alu6+17283)];
    var val13 = data2[(alu6+17284)];
    var val14 = data2[(alu6+17285)];
    var val15 = data2[(alu6+17286)];
    var val16 = data2[(alu6+17287)];
    var val17 = data2[(alu6+17288)];
    var val18 = data2[(alu6+34560)];
    var val19 = data2[(alu6+34561)];
    var val20 = data2[(alu6+34562)];
    var val21 = data2[(alu6+34563)];
    var val22 = data2[(alu6+34564)];
    var val23 = data2[(alu6+34565)];
    var val24 = data2[(alu6+34566)];
    var val25 = data2[(alu6+34567)];
    var val26 = data2[(alu6+34568)];
    var val27 = data2[(alu6+51840)];
    var val28 = data2[(alu6+51841)];
    var val29 = data2[(alu6+51842)];
    var val30 = data2[(alu6+51843)];
    var val31 = data2[(alu6+51844)];
    var val32 = data2[(alu6+51845)];
    var val33 = data2[(alu6+51846)];
    var val34 = data2[(alu6+51847)];
    var val35 = data2[(alu6+51848)];
    var alu7 = ((gidx1*491520)+(ridx0<<8)+alu0+alu1);
    var val36 = data1[alu7];
    var val37 = select((f16(0.0f)), data1[(alu7+-17)], (alu2&alu3));
    var val38 = select((f16(0.0f)), data1[(alu7+-16)], alu2);
    var val39 = select((f16(0.0f)), data1[(alu7+-15)], alu2);
    var val40 = select((f16(0.0f)), data1[(alu7+-14)], alu2);
    var val41 = select((f16(0.0f)), data1[(alu7+-13)], alu2);
    var val42 = select((f16(0.0f)), data1[(alu7+-12)], (alu5&alu2));
    var val43 = select((f16(0.0f)), data1[(alu7+-1)], alu3);
    var val44 = data1[(alu7+1)];
    var val45 = data1[(alu7+2)];
    var val46 = data1[(alu7+3)];
    var val47 = select((f16(0.0f)), data1[(alu7+4)], alu5);
    var val48 = select((f16(0.0f)), data1[(alu7+15)], (alu4&alu3));
    var val49 = select((f16(0.0f)), data1[(alu7+16)], alu4);
    var val50 = select((f16(0.0f)), data1[(alu7+17)], alu4);
    var val51 = select((f16(0.0f)), data1[(alu7+18)], alu4);
    var val52 = select((f16(0.0f)), data1[(alu7+19)], alu4);
    var val53 = select((f16(0.0f)), data1[(alu7+20)], (alu4&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu25 = ((gidx0<<3)+(lidx0<<2));
  var val54 = data3[alu25];
  var val55 = data4[alu25];
  var alu26 = (alu25+1);
  var val56 = data3[alu26];
  var val57 = data4[alu26];
  var alu27 = (alu25+2);
  var val58 = data3[alu27];
  var val59 = data4[alu27];
  var alu28 = (alu25+3);
  var val60 = data3[alu28];
  var val61 = data4[alu28];
  var alu29 = ((gidx0<<11)+(gidx1*327680)+(lidx0<<10)+alu0+alu1);
  data0[alu29] = (val55+(f16(acc0))+val54);
  data0[(alu29+1)] = (val55+(f16(acc4))+val54);
  data0[(alu29+2)] = (val55+(f16(acc8))+val54);
  data0[(alu29+3)] = (val55+(f16(acc12))+val54);
  data0[(alu29+256)] = (val57+(f16(acc1))+val56);
  data0[(alu29+257)] = (val57+(f16(acc5))+val56);
  data0[(alu29+258)] = (val57+(f16(acc9))+val56);
  data0[(alu29+259)] = (val57+(f16(acc13))+val56);
  data0[(alu29+512)] = (val59+(f16(acc2))+val58);
  data0[(alu29+513)] = (val59+(f16(acc6))+val58);
  data0[(alu29+514)] = (val59+(f16(acc10))+val58);
  data0[(alu29+515)] = (val59+(f16(acc14))+val58);
  data0[(alu29+768)] = (val61+(f16(acc3))+val60);
  data0[(alu29+769)] = (val61+(f16(acc7))+val60);
  data0[(alu29+770)] = (val61+(f16(acc11))+val60);
  data0[(alu29+771)] = (val61+(f16(acc15))+val60);
}`;

const r_2_40_4_8_16_480_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 40 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 480; ridx0++) {
    var alu2 = ((gidx1*61440)+(lidx0*7680)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+1920)];
    var val5 = data2[(alu2+1921)];
    var val6 = data2[(alu2+1922)];
    var val7 = data2[(alu2+1923)];
    var val8 = data2[(alu2+3840)];
    var val9 = data2[(alu2+3841)];
    var val10 = data2[(alu2+3842)];
    var val11 = data2[(alu2+3843)];
    var val12 = data2[(alu2+5760)];
    var val13 = data2[(alu2+5761)];
    var val14 = data2[(alu2+5762)];
    var val15 = data2[(alu2+5763)];
    var alu3 = (alu0+(gidx2*491520)+alu1+(ridx0<<10));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+256)];
    var val21 = data1[(alu3+257)];
    var val22 = data1[(alu3+258)];
    var val23 = data1[(alu3+259)];
    var val24 = data1[(alu3+512)];
    var val25 = data1[(alu3+513)];
    var val26 = data1[(alu3+514)];
    var val27 = data1[(alu3+515)];
    var val28 = data1[(alu3+768)];
    var val29 = data1[(alu3+769)];
    var val30 = data1[(alu3+770)];
    var val31 = data1[(alu3+771)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<13)+(gidx2*327680)+alu0+(lidx0<<10)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+256);
  var val44 = data4[alu29];
  var alu30 = (alu25+257);
  var val45 = data4[alu30];
  var alu31 = (alu25+258);
  var val46 = data4[alu31];
  var alu32 = (alu25+259);
  var val47 = data4[alu32];
  var alu33 = (alu25+512);
  var val48 = data4[alu33];
  var alu34 = (alu25+513);
  var val49 = data4[alu34];
  var alu35 = (alu25+514);
  var val50 = data4[alu35];
  var alu36 = (alu25+515);
  var val51 = data4[alu36];
  var alu37 = (alu25+768);
  var val52 = data4[alu37];
  var alu38 = (alu25+769);
  var val53 = data4[alu38];
  var alu39 = (alu25+770);
  var val54 = data4[alu39];
  var alu40 = (alu25+771);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const r_2_320_2_16_8_1280_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 320 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = ((lidx1<1)!=true);
  var alu1 = (((gidx0+lidx0)<1)!=true);
  var alu2 = ((lidx0+(gidx0<<4))<31);
  var alu3 = (lidx1<7);
  var alu4 = (lidx1<<1);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu5 = ((gidx1*46080)+(ridx0*9));
    var val0 = data2[alu5];
    var val1 = data2[(alu5+1)];
    var val2 = data2[(alu5+2)];
    var val3 = data2[(alu5+3)];
    var val4 = data2[(alu5+4)];
    var val5 = data2[(alu5+5)];
    var val6 = data2[(alu5+6)];
    var val7 = data2[(alu5+7)];
    var val8 = data2[(alu5+8)];
    var val9 = data2[(alu5+11520)];
    var val10 = data2[(alu5+11521)];
    var val11 = data2[(alu5+11522)];
    var val12 = data2[(alu5+11523)];
    var val13 = data2[(alu5+11524)];
    var val14 = data2[(alu5+11525)];
    var val15 = data2[(alu5+11526)];
    var val16 = data2[(alu5+11527)];
    var val17 = data2[(alu5+11528)];
    var val18 = data2[(alu5+23040)];
    var val19 = data2[(alu5+23041)];
    var val20 = data2[(alu5+23042)];
    var val21 = data2[(alu5+23043)];
    var val22 = data2[(alu5+23044)];
    var val23 = data2[(alu5+23045)];
    var val24 = data2[(alu5+23046)];
    var val25 = data2[(alu5+23047)];
    var val26 = data2[(alu5+23048)];
    var val27 = data2[(alu5+34560)];
    var val28 = data2[(alu5+34561)];
    var val29 = data2[(alu5+34562)];
    var val30 = data2[(alu5+34563)];
    var val31 = data2[(alu5+34564)];
    var val32 = data2[(alu5+34565)];
    var val33 = data2[(alu5+34566)];
    var val34 = data2[(alu5+34567)];
    var val35 = data2[(alu5+34568)];
    var alu6 = ((gidx0<<3)+(gidx2*20480)+(ridx0<<4)+((lidx0+1)>>1));
    var alu7 = (alu4+(((alu6+40959)%40960)<<4));
    var val36 = select((f16(0.0f)), data1[alu7], alu1);
    var val37 = select((f16(0.0f)), data1[(alu7+-1)], (alu0&alu1));
    var val38 = select((f16(0.0f)), data1[(alu7+1)], alu1);
    var val39 = select((f16(0.0f)), data1[(alu7+2)], (alu3&alu1));
    var alu8 = (alu4+((alu6%40960)<<4));
    var val40 = select((f16(0.0f)), data1[alu8], alu2);
    var val41 = select((f16(0.0f)), data1[(alu8+-1)], (alu2&alu0));
    var val42 = select((f16(0.0f)), data1[(alu8+1)], alu2);
    var val43 = select((f16(0.0f)), data1[(alu8+2)], (alu3&alu2));
    var alu9 = ((gidx0<<7)+(gidx2*327680)+(ridx0<<8)+((lidx0>>1)<<4)+alu4);
    var val44 = data1[alu9];
    var val45 = select((f16(0.0f)), data1[(alu9+-1)], alu0);
    var val46 = data1[(alu9+1)];
    var val47 = select((f16(0.0f)), data1[(alu9+2)], alu3);
    var cast0 = (f32((val46*val4)));
    var cast1 = (f32((val46*val5)));
    var cast2 = (f32((val46*val13)));
    var cast3 = (f32((val46*val14)));
    var cast4 = (f32((val46*val22)));
    var cast5 = (f32((val46*val23)));
    var cast6 = (f32((val46*val31)));
    var cast7 = (f32((val46*val32)));
    var cast8 = (f32((val44*val4)));
    var cast9 = (f32((val44*val13)));
    var cast10 = (f32((val44*val22)));
    var cast11 = (f32((val44*val31)));
    var cast12 = (f32((val38*val1)));
    var cast13 = (f32((val38*val2)));
    var cast14 = (f32((val38*val10)));
    var cast15 = (f32((val38*val11)));
    var cast16 = (f32((val38*val19)));
    var cast17 = (f32((val38*val20)));
    var cast18 = (f32((val38*val28)));
    var cast19 = (f32((val38*val29)));
    var cast20 = (f32((val42*val7)));
    var cast21 = (f32((val42*val8)));
    var cast22 = (f32((val42*val16)));
    var cast23 = (f32((val42*val17)));
    var cast24 = (f32((val42*val25)));
    var cast25 = (f32((val42*val26)));
    var cast26 = (f32((val42*val34)));
    var cast27 = (f32((val42*val35)));
    var cast28 = (f32((val36*val1)));
    var cast29 = (f32((val36*val10)));
    var cast30 = (f32((val36*val19)));
    var cast31 = (f32((val36*val28)));
    var alu10 = ((f32((val40*val6)))+(f32((val44*val3)))+(f32((val36*val0))));
    var cast32 = (f32((val40*val7)));
    var alu11 = ((f32((val40*val15)))+(f32((val44*val12)))+(f32((val36*val9))));
    var cast33 = (f32((val40*val16)));
    var alu12 = ((f32((val40*val24)))+(f32((val44*val21)))+(f32((val36*val18))));
    var cast34 = (f32((val40*val25)));
    var alu13 = ((f32((val40*val33)))+(f32((val44*val30)))+(f32((val36*val27))));
    var cast35 = (f32((val40*val34)));
    acc0 = (acc0+(f32((val40*val8)))+(f32((val44*val5)))+(f32((val36*val2)))+cast32+cast8+cast28+(f32((val41*val6)))+(f32((val45*val3)))+(f32((val37*val0))));
    acc1 = (acc1+(f32((val40*val17)))+(f32((val44*val14)))+(f32((val36*val11)))+cast33+cast9+cast29+(f32((val41*val15)))+(f32((val45*val12)))+(f32((val37*val9))));
    acc2 = (acc2+(f32((val40*val26)))+(f32((val44*val23)))+(f32((val36*val20)))+cast34+cast10+cast30+(f32((val41*val24)))+(f32((val45*val21)))+(f32((val37*val18))));
    acc3 = (acc3+(f32((val40*val35)))+(f32((val44*val32)))+(f32((val36*val29)))+cast35+cast11+cast31+(f32((val41*val33)))+(f32((val45*val30)))+(f32((val37*val27))));
    acc4 = (acc4+cast21+cast1+cast13+cast32+cast8+cast28+alu10);
    acc5 = (acc5+cast23+cast3+cast15+cast33+cast9+cast29+alu11);
    acc6 = (acc6+cast25+cast5+cast17+cast34+cast10+cast30+alu12);
    acc7 = (acc7+cast27+cast7+cast19+cast35+cast11+cast31+alu13);
    acc8 = (acc8+cast21+cast1+cast13+cast20+cast0+cast12+alu10);
    acc9 = (acc9+cast23+cast3+cast15+cast22+cast2+cast14+alu11);
    acc10 = (acc10+cast25+cast5+cast17+cast24+cast4+cast16+alu12);
    acc11 = (acc11+cast27+cast7+cast19+cast26+cast6+cast18+alu13);
    acc12 = (acc12+(f32((val43*val8)))+(f32((val47*val5)))+(f32((val39*val2)))+cast20+cast0+cast12+(f32((val42*val6)))+(f32((val46*val3)))+(f32((val38*val0))));
    acc13 = (acc13+(f32((val43*val17)))+(f32((val47*val14)))+(f32((val39*val11)))+cast22+cast2+cast14+(f32((val42*val15)))+(f32((val46*val12)))+(f32((val38*val9))));
    acc14 = (acc14+(f32((val43*val26)))+(f32((val47*val23)))+(f32((val39*val20)))+cast24+cast4+cast16+(f32((val42*val24)))+(f32((val46*val21)))+(f32((val38*val18))));
    acc15 = (acc15+(f32((val43*val35)))+(f32((val47*val32)))+(f32((val39*val29)))+cast26+cast6+cast18+(f32((val42*val33)))+(f32((val46*val30)))+(f32((val38*val27))));
  }
  var alu31 = (gidx1<<2);
  var val48 = data3[alu31];
  var val49 = data3[(alu31+1)];
  var val50 = data3[(alu31+2)];
  var val51 = data3[(alu31+3)];
  var alu32 = ((gidx1<<12)+(gidx2*1310720)+(gidx0<<9)+(lidx0<<5)+(lidx1<<2));
  data0[alu32] = ((f16(acc0))+val48);
  data0[(alu32+1)] = ((f16(acc4))+val48);
  data0[(alu32+2)] = ((f16(acc8))+val48);
  data0[(alu32+3)] = ((f16(acc12))+val48);
  data0[(alu32+1024)] = ((f16(acc1))+val49);
  data0[(alu32+1025)] = ((f16(acc5))+val49);
  data0[(alu32+1026)] = ((f16(acc9))+val49);
  data0[(alu32+1027)] = ((f16(acc13))+val49);
  data0[(alu32+2048)] = ((f16(acc2))+val50);
  data0[(alu32+2049)] = ((f16(acc6))+val50);
  data0[(alu32+2050)] = ((f16(acc10))+val50);
  data0[(alu32+2051)] = ((f16(acc14))+val50);
  data0[(alu32+3072)] = ((f16(acc3))+val51);
  data0[(alu32+3073)] = ((f16(acc7))+val51);
  data0[(alu32+3074)] = ((f16(acc11))+val51);
  data0[(alu32+3075)] = ((f16(acc15))+val51);
}`;

const E_2_240_16_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 240 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<160);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<13);
  var alu4 = (lidx0<<10);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*655360)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data2[(alu6+-1310720)], alu1);
  var val1 = select((f16(0.0f)), data2[(alu6+-1310719)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-1310718)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-1310717)], alu1);
  var alu7 = (alu3+(gidx2*1310720)+alu2+alu4+alu5);
  var val4 = select((f16(0.0f)), data1[alu7], alu0);
  var val5 = select((f16(0.0f)), data1[(alu7+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu7+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu7+3)], alu0);
  var alu8 = (alu3+(gidx2*1966080)+alu2+alu4+alu5);
  data0[alu8] = (val4+val0);
  data0[(alu8+1)] = (val5+val1);
  data0[(alu8+2)] = (val6+val2);
  data0[(alu8+3)] = (val7+val3);
}`;

const r_512_32_60_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 60; ridx0++) {
    var alu0 = ((gidx0*7680)+(lidx0*240)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+(f32(val3))+(f32(val2))+(f32(val1))+(f32(val0)));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_64_16_16n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = (f16((acc1*1.627604251552839e-05f)));
  }
}`;

const r_8_4_8_16_60_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 60; ridx0++) {
    var alu0 = ((gidx0*15360)+(gidx1*491520)+(lidx0*61440)+(lidx1*960)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+240)];
    var val6 = data1[(alu0+241)];
    var val7 = data1[(alu0+242)];
    var val8 = data1[(alu0+243)];
    var val9 = data1[(alu0+480)];
    var val10 = data1[(alu0+481)];
    var val11 = data1[(alu0+482)];
    var val12 = data1[(alu0+483)];
    var val13 = data1[(alu0+720)];
    var val14 = data1[(alu0+721)];
    var val15 = data1[(alu0+722)];
    var val16 = data1[(alu0+723)];
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
    acc0 = (acc0+(f32((alu3*alu3)))+(f32((alu2*alu2)))+(f32((alu1*alu1)))+(f32((alu16*alu16))));
    acc1 = (acc1+(f32((alu7*alu7)))+(f32((alu6*alu6)))+(f32((alu4*alu4)))+(f32((alu5*alu5))));
    acc2 = (acc2+(f32((alu11*alu11)))+(f32((alu10*alu10)))+(f32((alu8*alu8)))+(f32((alu9*alu9))));
    acc3 = (acc3+(f32((alu15*alu15)))+(f32((alu14*alu14)))+(f32((alu12*alu12)))+(f32((alu13*alu13))));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_64_16_16n3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = sqrt((1/((f16((acc1*1.627604251552839e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_240_16_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 240 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/60));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<13)+(gidx2*1966080)+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_8_1920_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<5);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<4))<31);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 1920; ridx0++) {
    var alu7 = ((gidx1*69120)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+17280)];
    var val10 = data2[(alu7+17281)];
    var val11 = data2[(alu7+17282)];
    var val12 = data2[(alu7+17283)];
    var val13 = data2[(alu7+17284)];
    var val14 = data2[(alu7+17285)];
    var val15 = data2[(alu7+17286)];
    var val16 = data2[(alu7+17287)];
    var val17 = data2[(alu7+17288)];
    var val18 = data2[(alu7+34560)];
    var val19 = data2[(alu7+34561)];
    var val20 = data2[(alu7+34562)];
    var val21 = data2[(alu7+34563)];
    var val22 = data2[(alu7+34564)];
    var val23 = data2[(alu7+34565)];
    var val24 = data2[(alu7+34566)];
    var val25 = data2[(alu7+34567)];
    var val26 = data2[(alu7+34568)];
    var val27 = data2[(alu7+51840)];
    var val28 = data2[(alu7+51841)];
    var val29 = data2[(alu7+51842)];
    var val30 = data2[(alu7+51843)];
    var val31 = data2[(alu7+51844)];
    var val32 = data2[(alu7+51845)];
    var val33 = data2[(alu7+51846)];
    var val34 = data2[(alu7+51847)];
    var val35 = data2[(alu7+51848)];
    var alu8 = (alu0+alu1+(gidx2*1966080)+(ridx0<<10)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-33)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-32)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-31)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-30)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-29)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-28)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+31)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+32)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+33)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+34)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+35)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+36)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx1<<12)+(gidx2*655360)+alu0+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+1024)] = (val57+(f16(acc1))+val56);
  data0[(alu30+1025)] = (val57+(f16(acc5))+val56);
  data0[(alu30+1026)] = (val57+(f16(acc9))+val56);
  data0[(alu30+1027)] = (val57+(f16(acc13))+val56);
  data0[(alu30+2048)] = (val59+(f16(acc2))+val58);
  data0[(alu30+2049)] = (val59+(f16(acc6))+val58);
  data0[(alu30+2050)] = (val59+(f16(acc10))+val58);
  data0[(alu30+2051)] = (val59+(f16(acc14))+val58);
  data0[(alu30+3072)] = (val61+(f16(acc3))+val60);
  data0[(alu30+3073)] = (val61+(f16(acc7))+val60);
  data0[(alu30+3074)] = (val61+(f16(acc11))+val60);
  data0[(alu30+3075)] = (val61+(f16(acc15))+val60);
}`;

const r_2_20_16_8_16_480_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 20 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 480; ridx0++) {
    var alu2 = ((gidx1*61440)+(lidx0*7680)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+1920)];
    var val5 = data2[(alu2+1921)];
    var val6 = data2[(alu2+1922)];
    var val7 = data2[(alu2+1923)];
    var val8 = data2[(alu2+3840)];
    var val9 = data2[(alu2+3841)];
    var val10 = data2[(alu2+3842)];
    var val11 = data2[(alu2+3843)];
    var val12 = data2[(alu2+5760)];
    var val13 = data2[(alu2+5761)];
    var val14 = data2[(alu2+5762)];
    var val15 = data2[(alu2+5763)];
    var alu3 = (alu0+(gidx2*1966080)+alu1+(ridx0<<12));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+1024)];
    var val21 = data1[(alu3+1025)];
    var val22 = data1[(alu3+1026)];
    var val23 = data1[(alu3+1027)];
    var val24 = data1[(alu3+2048)];
    var val25 = data1[(alu3+2049)];
    var val26 = data1[(alu3+2050)];
    var val27 = data1[(alu3+2051)];
    var val28 = data1[(alu3+3072)];
    var val29 = data1[(alu3+3073)];
    var val30 = data1[(alu3+3074)];
    var val31 = data1[(alu3+3075)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<15)+(gidx2*655360)+alu0+(lidx0<<12)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+1024);
  var val44 = data4[alu29];
  var alu30 = (alu25+1025);
  var val45 = data4[alu30];
  var alu31 = (alu25+1026);
  var val46 = data4[alu31];
  var alu32 = (alu25+1027);
  var val47 = data4[alu32];
  var alu33 = (alu25+2048);
  var val48 = data4[alu33];
  var alu34 = (alu25+2049);
  var val49 = data4[alu34];
  var alu35 = (alu25+2050);
  var val50 = data4[alu35];
  var alu36 = (alu25+2051);
  var val51 = data4[alu36];
  var alu37 = (alu25+3072);
  var val52 = data4[alu37];
  var alu38 = (alu25+3073);
  var val53 = data4[alu38];
  var alu39 = (alu25+3074);
  var val54 = data4[alu39];
  var alu40 = (alu25+3075);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const E_2_160_16_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<80);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<13);
  var alu4 = (lidx0<<10);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*655360)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data1[alu6], alu0);
  var val1 = select((f16(0.0f)), data2[(alu6+-655360)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-655359)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-655358)], alu1);
  var val4 = select((f16(0.0f)), data2[(alu6+-655357)], alu1);
  var val5 = select((f16(0.0f)), data1[(alu6+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu6+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu6+3)], alu0);
  var alu7 = (alu3+(gidx2*1310720)+alu2+alu4+alu5);
  data0[alu7] = (val0+val1);
  data0[(alu7+1)] = (val5+val2);
  data0[(alu7+2)] = (val6+val3);
  data0[(alu7+3)] = (val7+val4);
}`;

const E_2_160_16_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(gidx1/5));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<13)+(gidx2*1310720)+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_8_1280_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<5);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<4))<31);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 1280; ridx0++) {
    var alu7 = ((gidx1*46080)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+11520)];
    var val10 = data2[(alu7+11521)];
    var val11 = data2[(alu7+11522)];
    var val12 = data2[(alu7+11523)];
    var val13 = data2[(alu7+11524)];
    var val14 = data2[(alu7+11525)];
    var val15 = data2[(alu7+11526)];
    var val16 = data2[(alu7+11527)];
    var val17 = data2[(alu7+11528)];
    var val18 = data2[(alu7+23040)];
    var val19 = data2[(alu7+23041)];
    var val20 = data2[(alu7+23042)];
    var val21 = data2[(alu7+23043)];
    var val22 = data2[(alu7+23044)];
    var val23 = data2[(alu7+23045)];
    var val24 = data2[(alu7+23046)];
    var val25 = data2[(alu7+23047)];
    var val26 = data2[(alu7+23048)];
    var val27 = data2[(alu7+34560)];
    var val28 = data2[(alu7+34561)];
    var val29 = data2[(alu7+34562)];
    var val30 = data2[(alu7+34563)];
    var val31 = data2[(alu7+34564)];
    var val32 = data2[(alu7+34565)];
    var val33 = data2[(alu7+34566)];
    var val34 = data2[(alu7+34567)];
    var val35 = data2[(alu7+34568)];
    var alu8 = (alu0+alu1+(gidx2*1310720)+(ridx0<<10)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-33)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-32)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-31)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-30)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-29)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-28)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+31)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+32)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+33)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+34)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+35)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+36)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx1<<12)+(gidx2*655360)+alu0+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+1024)] = (val57+(f16(acc1))+val56);
  data0[(alu30+1025)] = (val57+(f16(acc5))+val56);
  data0[(alu30+1026)] = (val57+(f16(acc9))+val56);
  data0[(alu30+1027)] = (val57+(f16(acc13))+val56);
  data0[(alu30+2048)] = (val59+(f16(acc2))+val58);
  data0[(alu30+2049)] = (val59+(f16(acc6))+val58);
  data0[(alu30+2050)] = (val59+(f16(acc10))+val58);
  data0[(alu30+2051)] = (val59+(f16(acc14))+val58);
  data0[(alu30+3072)] = (val61+(f16(acc3))+val60);
  data0[(alu30+3073)] = (val61+(f16(acc7))+val60);
  data0[(alu30+3074)] = (val61+(f16(acc11))+val60);
  data0[(alu30+3075)] = (val61+(f16(acc15))+val60);
}`;

const r_2_20_16_8_16_320_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 20 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu2 = ((gidx1*40960)+(lidx0*5120)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+1280)];
    var val5 = data2[(alu2+1281)];
    var val6 = data2[(alu2+1282)];
    var val7 = data2[(alu2+1283)];
    var val8 = data2[(alu2+2560)];
    var val9 = data2[(alu2+2561)];
    var val10 = data2[(alu2+2562)];
    var val11 = data2[(alu2+2563)];
    var val12 = data2[(alu2+3840)];
    var val13 = data2[(alu2+3841)];
    var val14 = data2[(alu2+3842)];
    var val15 = data2[(alu2+3843)];
    var alu3 = (alu0+(gidx2*1310720)+alu1+(ridx0<<12));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+1024)];
    var val21 = data1[(alu3+1025)];
    var val22 = data1[(alu3+1026)];
    var val23 = data1[(alu3+1027)];
    var val24 = data1[(alu3+2048)];
    var val25 = data1[(alu3+2049)];
    var val26 = data1[(alu3+2050)];
    var val27 = data1[(alu3+2051)];
    var val28 = data1[(alu3+3072)];
    var val29 = data1[(alu3+3073)];
    var val30 = data1[(alu3+3074)];
    var val31 = data1[(alu3+3075)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<15)+(gidx2*655360)+alu0+(lidx0<<12)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+1024);
  var val44 = data4[alu29];
  var alu30 = (alu25+1025);
  var val45 = data4[alu30];
  var alu31 = (alu25+1026);
  var val46 = data4[alu31];
  var alu32 = (alu25+1027);
  var val47 = data4[alu32];
  var alu33 = (alu25+2048);
  var val48 = data4[alu33];
  var alu34 = (alu25+2049);
  var val49 = data4[alu34];
  var alu35 = (alu25+2050);
  var val50 = data4[alu35];
  var alu36 = (alu25+2051);
  var val51 = data4[alu36];
  var alu37 = (alu25+3072);
  var val52 = data4[alu37];
  var alu38 = (alu25+3073);
  var val53 = data4[alu38];
  var alu39 = (alu25+3074);
  var val54 = data4[alu39];
  var alu40 = (alu25+3075);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const E_2_120_16_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 120 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<80);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<13);
  var alu4 = (lidx0<<10);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*327680)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data2[(alu6+-655360)], alu1);
  var val1 = select((f16(0.0f)), data2[(alu6+-655359)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-655358)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-655357)], alu1);
  var alu7 = (alu3+(gidx2*655360)+alu2+alu4+alu5);
  var val4 = select((f16(0.0f)), data1[alu7], alu0);
  var val5 = select((f16(0.0f)), data1[(alu7+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu7+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu7+3)], alu0);
  var alu8 = (alu3+(gidx2*983040)+alu2+alu4+alu5);
  data0[alu8] = (val4+val0);
  data0[(alu8+1)] = (val5+val1);
  data0[(alu8+2)] = (val6+val2);
  data0[(alu8+3)] = (val7+val3);
}`;

const r_64_16_1920 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1920; ridx0++) {
    var val0 = data1[((gidx0*30720)+(lidx0*1920)+ridx0)];
    acc0 = (acc0+(f32(val0)));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = (f16((acc1*3.255208503105678e-05f)));
  }
}`;

const r_64_16_1920n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data2[gidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1920; ridx0++) {
    var val1 = data1[((gidx0*30720)+(lidx0*1920)+ridx0)];
    var alu0 = (val1-val0);
    acc0 = (acc0+(f32((alu0*alu0))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((f16((acc1*3.255208503105678e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_120_16_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 120 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/30));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<13)+(gidx2*983040)+(gidx0<<6)+(lidx0<<10)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_160_2_16_8_960_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx0<<9);
  var alu1 = (lidx0<<5);
  var alu2 = (lidx1<<2);
  var alu3 = ((lidx1<1)!=true);
  var alu4 = (((gidx0+lidx0)<1)!=true);
  var alu5 = ((lidx0+(gidx0<<4))<31);
  var alu6 = (lidx1<7);
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
  for (var ridx0 = 0; ridx0 < 960; ridx0++) {
    var alu7 = ((gidx1*34560)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+8640)];
    var val10 = data2[(alu7+8641)];
    var val11 = data2[(alu7+8642)];
    var val12 = data2[(alu7+8643)];
    var val13 = data2[(alu7+8644)];
    var val14 = data2[(alu7+8645)];
    var val15 = data2[(alu7+8646)];
    var val16 = data2[(alu7+8647)];
    var val17 = data2[(alu7+8648)];
    var val18 = data2[(alu7+17280)];
    var val19 = data2[(alu7+17281)];
    var val20 = data2[(alu7+17282)];
    var val21 = data2[(alu7+17283)];
    var val22 = data2[(alu7+17284)];
    var val23 = data2[(alu7+17285)];
    var val24 = data2[(alu7+17286)];
    var val25 = data2[(alu7+17287)];
    var val26 = data2[(alu7+17288)];
    var val27 = data2[(alu7+25920)];
    var val28 = data2[(alu7+25921)];
    var val29 = data2[(alu7+25922)];
    var val30 = data2[(alu7+25923)];
    var val31 = data2[(alu7+25924)];
    var val32 = data2[(alu7+25925)];
    var val33 = data2[(alu7+25926)];
    var val34 = data2[(alu7+25927)];
    var val35 = data2[(alu7+25928)];
    var alu8 = (alu0+alu1+(gidx2*983040)+(ridx0<<10)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-33)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-32)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-31)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-30)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-29)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-28)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+31)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+32)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+33)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+34)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+35)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+36)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx1<<12)+(gidx2*655360)+alu0+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+1024)] = (val57+(f16(acc1))+val56);
  data0[(alu30+1025)] = (val57+(f16(acc5))+val56);
  data0[(alu30+1026)] = (val57+(f16(acc9))+val56);
  data0[(alu30+1027)] = (val57+(f16(acc13))+val56);
  data0[(alu30+2048)] = (val59+(f16(acc2))+val58);
  data0[(alu30+2049)] = (val59+(f16(acc6))+val58);
  data0[(alu30+2050)] = (val59+(f16(acc10))+val58);
  data0[(alu30+2051)] = (val59+(f16(acc14))+val58);
  data0[(alu30+3072)] = (val61+(f16(acc3))+val60);
  data0[(alu30+3073)] = (val61+(f16(acc7))+val60);
  data0[(alu30+3074)] = (val61+(f16(acc11))+val60);
  data0[(alu30+3075)] = (val61+(f16(acc15))+val60);
}`;

const r_2_20_16_8_16_240_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 20 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 240; ridx0++) {
    var alu2 = ((gidx1*30720)+(lidx0*3840)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+960)];
    var val5 = data2[(alu2+961)];
    var val6 = data2[(alu2+962)];
    var val7 = data2[(alu2+963)];
    var val8 = data2[(alu2+1920)];
    var val9 = data2[(alu2+1921)];
    var val10 = data2[(alu2+1922)];
    var val11 = data2[(alu2+1923)];
    var val12 = data2[(alu2+2880)];
    var val13 = data2[(alu2+2881)];
    var val14 = data2[(alu2+2882)];
    var val15 = data2[(alu2+2883)];
    var alu3 = (alu0+(gidx2*983040)+alu1+(ridx0<<12));
    var val16 = data1[alu3];
    var val17 = data1[(alu3+1)];
    var val18 = data1[(alu3+2)];
    var val19 = data1[(alu3+3)];
    var val20 = data1[(alu3+1024)];
    var val21 = data1[(alu3+1025)];
    var val22 = data1[(alu3+1026)];
    var val23 = data1[(alu3+1027)];
    var val24 = data1[(alu3+2048)];
    var val25 = data1[(alu3+2049)];
    var val26 = data1[(alu3+2050)];
    var val27 = data1[(alu3+2051)];
    var val28 = data1[(alu3+3072)];
    var val29 = data1[(alu3+3073)];
    var val30 = data1[(alu3+3074)];
    var val31 = data1[(alu3+3075)];
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<15)+(gidx2*655360)+alu0+(lidx0<<12)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+1024);
  var val44 = data4[alu29];
  var alu30 = (alu25+1025);
  var val45 = data4[alu30];
  var alu31 = (alu25+1026);
  var val46 = data4[alu31];
  var alu32 = (alu25+1027);
  var val47 = data4[alu32];
  var alu33 = (alu25+2048);
  var val48 = data4[alu33];
  var alu34 = (alu25+2049);
  var val49 = data4[alu34];
  var alu35 = (alu25+2050);
  var val50 = data4[alu35];
  var alu36 = (alu25+2051);
  var val51 = data4[alu36];
  var alu37 = (alu25+3072);
  var val52 = data4[alu37];
  var alu38 = (alu25+3073);
  var val53 = data4[alu38];
  var alu39 = (alu25+3074);
  var val54 = data4[alu39];
  var alu40 = (alu25+3075);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const r_2_160_8_8_16_640_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 160 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = ((lidx0+1)>>1);
  var alu1 = ((lidx1<1)!=true);
  var alu2 = (((gidx0+lidx0)<1)!=true);
  var alu3 = ((lidx0+(gidx0<<3))<63);
  var alu4 = (lidx1<15);
  var alu5 = (lidx1<<1);
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu6 = ((gidx1*23040)+(ridx0*9));
    var val0 = data2[alu6];
    var val1 = data2[(alu6+1)];
    var val2 = data2[(alu6+2)];
    var val3 = data2[(alu6+3)];
    var val4 = data2[(alu6+4)];
    var val5 = data2[(alu6+5)];
    var val6 = data2[(alu6+6)];
    var val7 = data2[(alu6+7)];
    var val8 = data2[(alu6+8)];
    var val9 = data2[(alu6+5760)];
    var val10 = data2[(alu6+5761)];
    var val11 = data2[(alu6+5762)];
    var val12 = data2[(alu6+5763)];
    var val13 = data2[(alu6+5764)];
    var val14 = data2[(alu6+5765)];
    var val15 = data2[(alu6+5766)];
    var val16 = data2[(alu6+5767)];
    var val17 = data2[(alu6+5768)];
    var val18 = data2[(alu6+11520)];
    var val19 = data2[(alu6+11521)];
    var val20 = data2[(alu6+11522)];
    var val21 = data2[(alu6+11523)];
    var val22 = data2[(alu6+11524)];
    var val23 = data2[(alu6+11525)];
    var val24 = data2[(alu6+11526)];
    var val25 = data2[(alu6+11527)];
    var val26 = data2[(alu6+11528)];
    var val27 = data2[(alu6+17280)];
    var val28 = data2[(alu6+17281)];
    var val29 = data2[(alu6+17282)];
    var val30 = data2[(alu6+17283)];
    var val31 = data2[(alu6+17284)];
    var val32 = data2[(alu6+17285)];
    var val33 = data2[(alu6+17286)];
    var val34 = data2[(alu6+17287)];
    var val35 = data2[(alu6+17288)];
    var alu7 = (alu5+((((gidx0<<2)+(gidx2*20480)+(ridx0<<5)+alu0)%40960)<<5));
    var val36 = select((f16(0.0f)), data1[alu7], alu3);
    var val37 = select((f16(0.0f)), data1[(alu7+-1)], (alu3&alu1));
    var val38 = select((f16(0.0f)), data1[(alu7+1)], alu3);
    var val39 = select((f16(0.0f)), data1[(alu7+2)], (alu4&alu3));
    var alu8 = ((gidx0<<7)+(gidx2*655360)+(ridx0<<10));
    var alu9 = (alu8+((lidx0>>1)<<5)+alu5);
    var val40 = data1[alu9];
    var val41 = select((f16(0.0f)), data1[(alu9+-1)], alu1);
    var val42 = data1[(alu9+1)];
    var val43 = select((f16(0.0f)), data1[(alu9+2)], alu4);
    var alu10 = (alu8+(alu0<<5)+alu5);
    var val44 = select((f16(0.0f)), data1[(alu10+-33)], (alu1&alu2));
    var val45 = select((f16(0.0f)), data1[(alu10+-32)], alu2);
    var val46 = select((f16(0.0f)), data1[(alu10+-31)], alu2);
    var val47 = select((f16(0.0f)), data1[(alu10+-30)], (alu4&alu2));
    var cast0 = (f32((val42*val4)));
    var cast1 = (f32((val42*val5)));
    var cast2 = (f32((val42*val13)));
    var cast3 = (f32((val42*val14)));
    var cast4 = (f32((val42*val22)));
    var cast5 = (f32((val42*val23)));
    var cast6 = (f32((val42*val31)));
    var cast7 = (f32((val42*val32)));
    var cast8 = (f32((val45*val1)));
    var cast9 = (f32((val45*val10)));
    var cast10 = (f32((val45*val19)));
    var cast11 = (f32((val45*val28)));
    var cast12 = (f32((val46*val1)));
    var cast13 = (f32((val46*val2)));
    var cast14 = (f32((val46*val10)));
    var cast15 = (f32((val46*val11)));
    var cast16 = (f32((val46*val19)));
    var cast17 = (f32((val46*val20)));
    var cast18 = (f32((val46*val28)));
    var cast19 = (f32((val46*val29)));
    var cast20 = (f32((val40*val4)));
    var cast21 = (f32((val40*val13)));
    var cast22 = (f32((val40*val22)));
    var cast23 = (f32((val40*val31)));
    var cast24 = (f32((val38*val7)));
    var cast25 = (f32((val38*val8)));
    var cast26 = (f32((val38*val16)));
    var cast27 = (f32((val38*val17)));
    var cast28 = (f32((val38*val25)));
    var cast29 = (f32((val38*val26)));
    var cast30 = (f32((val38*val34)));
    var cast31 = (f32((val38*val35)));
    var alu11 = ((f32((val36*val6)))+(f32((val45*val0)))+(f32((val40*val3))));
    var cast32 = (f32((val36*val7)));
    var alu12 = ((f32((val36*val15)))+(f32((val45*val9)))+(f32((val40*val12))));
    var cast33 = (f32((val36*val16)));
    var alu13 = ((f32((val36*val24)))+(f32((val45*val18)))+(f32((val40*val21))));
    var cast34 = (f32((val36*val25)));
    var alu14 = ((f32((val36*val33)))+(f32((val45*val27)))+(f32((val40*val30))));
    var cast35 = (f32((val36*val34)));
    acc0 = (acc0+(f32((val36*val8)))+(f32((val40*val5)))+(f32((val45*val2)))+cast32+cast20+cast8+(f32((val37*val6)))+(f32((val41*val3)))+(f32((val44*val0))));
    acc1 = (acc1+(f32((val36*val17)))+(f32((val40*val14)))+(f32((val45*val11)))+cast33+cast21+cast9+(f32((val37*val15)))+(f32((val41*val12)))+(f32((val44*val9))));
    acc2 = (acc2+(f32((val36*val26)))+(f32((val40*val23)))+(f32((val45*val20)))+cast34+cast22+cast10+(f32((val37*val24)))+(f32((val41*val21)))+(f32((val44*val18))));
    acc3 = (acc3+(f32((val36*val35)))+(f32((val40*val32)))+(f32((val45*val29)))+cast35+cast23+cast11+(f32((val37*val33)))+(f32((val41*val30)))+(f32((val44*val27))));
    acc4 = (acc4+cast25+cast1+cast13+cast32+cast20+cast8+alu11);
    acc5 = (acc5+cast27+cast3+cast15+cast33+cast21+cast9+alu12);
    acc6 = (acc6+cast29+cast5+cast17+cast34+cast22+cast10+alu13);
    acc7 = (acc7+cast31+cast7+cast19+cast35+cast23+cast11+alu14);
    acc8 = (acc8+cast25+cast1+cast13+cast24+cast0+cast12+alu11);
    acc9 = (acc9+cast27+cast3+cast15+cast26+cast2+cast14+alu12);
    acc10 = (acc10+cast29+cast5+cast17+cast28+cast4+cast16+alu13);
    acc11 = (acc11+cast31+cast7+cast19+cast30+cast6+cast18+alu14);
    acc12 = (acc12+(f32((val39*val8)))+(f32((val43*val5)))+(f32((val47*val2)))+cast24+cast0+cast12+(f32((val38*val6)))+(f32((val42*val3)))+(f32((val46*val0))));
    acc13 = (acc13+(f32((val39*val17)))+(f32((val43*val14)))+(f32((val47*val11)))+cast26+cast2+cast14+(f32((val38*val15)))+(f32((val42*val12)))+(f32((val46*val9))));
    acc14 = (acc14+(f32((val39*val26)))+(f32((val43*val23)))+(f32((val47*val20)))+cast28+cast4+cast16+(f32((val38*val24)))+(f32((val42*val21)))+(f32((val46*val18))));
    acc15 = (acc15+(f32((val39*val35)))+(f32((val43*val32)))+(f32((val47*val29)))+cast30+cast6+cast18+(f32((val38*val33)))+(f32((val42*val30)))+(f32((val46*val27))));
  }
  var alu32 = (gidx1<<2);
  var val48 = data3[alu32];
  var val49 = data3[(alu32+1)];
  var val50 = data3[(alu32+2)];
  var val51 = data3[(alu32+3)];
  var alu33 = ((gidx1<<14)+(gidx2*2621440)+(gidx0<<9)+(lidx0<<6)+(lidx1<<2));
  data0[alu33] = ((f16(acc0))+val48);
  data0[(alu33+1)] = ((f16(acc4))+val48);
  data0[(alu33+2)] = ((f16(acc8))+val48);
  data0[(alu33+3)] = ((f16(acc12))+val48);
  data0[(alu33+4096)] = ((f16(acc1))+val49);
  data0[(alu33+4097)] = ((f16(acc5))+val49);
  data0[(alu33+4098)] = ((f16(acc9))+val49);
  data0[(alu33+4099)] = ((f16(acc13))+val49);
  data0[(alu33+8192)] = ((f16(acc2))+val50);
  data0[(alu33+8193)] = ((f16(acc6))+val50);
  data0[(alu33+8194)] = ((f16(acc10))+val50);
  data0[(alu33+8195)] = ((f16(acc14))+val50);
  data0[(alu33+12288)] = ((f16(acc3))+val51);
  data0[(alu33+12289)] = ((f16(acc7))+val51);
  data0[(alu33+12290)] = ((f16(acc11))+val51);
  data0[(alu33+12291)] = ((f16(acc15))+val51);
}`;

const E_2_120_64_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 120 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<80);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<15);
  var alu4 = (lidx0<<12);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*1310720)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data2[(alu6+-2621440)], alu1);
  var val1 = select((f16(0.0f)), data2[(alu6+-2621439)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-2621438)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-2621437)], alu1);
  var alu7 = (alu3+(gidx2*2621440)+alu2+alu4+alu5);
  var val4 = select((f16(0.0f)), data1[alu7], alu0);
  var val5 = select((f16(0.0f)), data1[(alu7+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu7+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu7+3)], alu0);
  var alu8 = (alu3+(gidx2*3932160)+alu2+alu4+alu5);
  data0[alu8] = (val4+val0);
  data0[(alu8+1)] = (val5+val1);
  data0[(alu8+2)] = (val6+val2);
  data0[(alu8+3)] = (val7+val3);
}`;

const r_512_32_120_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 120; ridx0++) {
    var alu0 = ((gidx0*15360)+(lidx0*480)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+(f32(val3))+(f32(val2))+(f32(val1))+(f32(val0)));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_64_16_16n4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = (f16((acc1*8.138021257764194e-06f)));
  }
}`;

const r_8_4_8_16_120_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 120; ridx0++) {
    var alu0 = ((gidx0*30720)+(gidx1*983040)+(lidx0*122880)+(lidx1*1920)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+480)];
    var val6 = data1[(alu0+481)];
    var val7 = data1[(alu0+482)];
    var val8 = data1[(alu0+483)];
    var val9 = data1[(alu0+960)];
    var val10 = data1[(alu0+961)];
    var val11 = data1[(alu0+962)];
    var val12 = data1[(alu0+963)];
    var val13 = data1[(alu0+1440)];
    var val14 = data1[(alu0+1441)];
    var val15 = data1[(alu0+1442)];
    var val16 = data1[(alu0+1443)];
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
    acc0 = (acc0+(f32((alu3*alu3)))+(f32((alu2*alu2)))+(f32((alu1*alu1)))+(f32((alu16*alu16))));
    acc1 = (acc1+(f32((alu7*alu7)))+(f32((alu6*alu6)))+(f32((alu4*alu4)))+(f32((alu5*alu5))));
    acc2 = (acc2+(f32((alu11*alu11)))+(f32((alu10*alu10)))+(f32((alu8*alu8)))+(f32((alu9*alu9))));
    acc3 = (acc3+(f32((alu15*alu15)))+(f32((alu14*alu14)))+(f32((alu12*alu12)))+(f32((alu13*alu13))));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_64_16_16n5 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = sqrt((1/((f16((acc1*8.138021257764194e-06f)))+(f16(1e-05f)))));
  }
}`;

const E_2_120_64_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 120 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/30));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<15)+(gidx2*3932160)+(gidx0<<6)+(lidx0<<12)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_80_8_8_16_960_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 960; ridx0++) {
    var alu7 = ((gidx1*34560)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+8640)];
    var val10 = data2[(alu7+8641)];
    var val11 = data2[(alu7+8642)];
    var val12 = data2[(alu7+8643)];
    var val13 = data2[(alu7+8644)];
    var val14 = data2[(alu7+8645)];
    var val15 = data2[(alu7+8646)];
    var val16 = data2[(alu7+8647)];
    var val17 = data2[(alu7+8648)];
    var val18 = data2[(alu7+17280)];
    var val19 = data2[(alu7+17281)];
    var val20 = data2[(alu7+17282)];
    var val21 = data2[(alu7+17283)];
    var val22 = data2[(alu7+17284)];
    var val23 = data2[(alu7+17285)];
    var val24 = data2[(alu7+17286)];
    var val25 = data2[(alu7+17287)];
    var val26 = data2[(alu7+17288)];
    var val27 = data2[(alu7+25920)];
    var val28 = data2[(alu7+25921)];
    var val29 = data2[(alu7+25922)];
    var val30 = data2[(alu7+25923)];
    var val31 = data2[(alu7+25924)];
    var val32 = data2[(alu7+25925)];
    var val33 = data2[(alu7+25926)];
    var val34 = data2[(alu7+25927)];
    var val35 = data2[(alu7+25928)];
    var alu8 = (alu0+alu1+(gidx2*3932160)+(ridx0<<12)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-65)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-64)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-63)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-62)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-61)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-60)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+63)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+64)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+65)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+66)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+67)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx1<<14)+(gidx2*1310720)+alu0+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+4096)] = (val57+(f16(acc1))+val56);
  data0[(alu30+4097)] = (val57+(f16(acc5))+val56);
  data0[(alu30+4098)] = (val57+(f16(acc9))+val56);
  data0[(alu30+4099)] = (val57+(f16(acc13))+val56);
  data0[(alu30+8192)] = (val59+(f16(acc2))+val58);
  data0[(alu30+8193)] = (val59+(f16(acc6))+val58);
  data0[(alu30+8194)] = (val59+(f16(acc10))+val58);
  data0[(alu30+8195)] = (val59+(f16(acc14))+val58);
  data0[(alu30+12288)] = (val61+(f16(acc3))+val60);
  data0[(alu30+12289)] = (val61+(f16(acc7))+val60);
  data0[(alu30+12290)] = (val61+(f16(acc11))+val60);
  data0[(alu30+12291)] = (val61+(f16(acc15))+val60);
}`;

const r_2_80_8_8_16_320_4_4_3_3n2 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx2*1310720);
  var alu1 = (gidx0<<9);
  var alu2 = (lidx0<<6);
  var alu3 = (lidx1<<2);
  var alu4 = ((lidx1<1)!=true);
  var alu5 = (((gidx0+lidx0)<1)!=true);
  var alu6 = ((lidx0+(gidx0<<3))<63);
  var alu7 = (lidx1<15);
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu8 = ((gidx1*11520)+(ridx0*9));
    var val0 = data2[alu8];
    var val1 = data2[(alu8+1)];
    var val2 = data2[(alu8+2)];
    var val3 = data2[(alu8+3)];
    var val4 = data2[(alu8+4)];
    var val5 = data2[(alu8+5)];
    var val6 = data2[(alu8+6)];
    var val7 = data2[(alu8+7)];
    var val8 = data2[(alu8+8)];
    var val9 = data2[(alu8+2880)];
    var val10 = data2[(alu8+2881)];
    var val11 = data2[(alu8+2882)];
    var val12 = data2[(alu8+2883)];
    var val13 = data2[(alu8+2884)];
    var val14 = data2[(alu8+2885)];
    var val15 = data2[(alu8+2886)];
    var val16 = data2[(alu8+2887)];
    var val17 = data2[(alu8+2888)];
    var val18 = data2[(alu8+5760)];
    var val19 = data2[(alu8+5761)];
    var val20 = data2[(alu8+5762)];
    var val21 = data2[(alu8+5763)];
    var val22 = data2[(alu8+5764)];
    var val23 = data2[(alu8+5765)];
    var val24 = data2[(alu8+5766)];
    var val25 = data2[(alu8+5767)];
    var val26 = data2[(alu8+5768)];
    var val27 = data2[(alu8+8640)];
    var val28 = data2[(alu8+8641)];
    var val29 = data2[(alu8+8642)];
    var val30 = data2[(alu8+8643)];
    var val31 = data2[(alu8+8644)];
    var val32 = data2[(alu8+8645)];
    var val33 = data2[(alu8+8646)];
    var val34 = data2[(alu8+8647)];
    var val35 = data2[(alu8+8648)];
    var alu9 = (alu1+alu2+alu0+(ridx0<<12)+alu3);
    var val36 = data1[alu9];
    var val37 = select((f16(0.0f)), data1[(alu9+-65)], (alu4&alu5));
    var val38 = select((f16(0.0f)), data1[(alu9+-64)], alu5);
    var val39 = select((f16(0.0f)), data1[(alu9+-63)], alu5);
    var val40 = select((f16(0.0f)), data1[(alu9+-62)], alu5);
    var val41 = select((f16(0.0f)), data1[(alu9+-61)], alu5);
    var val42 = select((f16(0.0f)), data1[(alu9+-60)], (alu7&alu5));
    var val43 = select((f16(0.0f)), data1[(alu9+-1)], alu4);
    var val44 = data1[(alu9+1)];
    var val45 = data1[(alu9+2)];
    var val46 = data1[(alu9+3)];
    var val47 = select((f16(0.0f)), data1[(alu9+4)], alu7);
    var val48 = select((f16(0.0f)), data1[(alu9+63)], (alu6&alu4));
    var val49 = select((f16(0.0f)), data1[(alu9+64)], alu6);
    var val50 = select((f16(0.0f)), data1[(alu9+65)], alu6);
    var val51 = select((f16(0.0f)), data1[(alu9+66)], alu6);
    var val52 = select((f16(0.0f)), data1[(alu9+67)], alu6);
    var val53 = select((f16(0.0f)), data1[(alu9+68)], (alu7&alu6));
    acc0 = (acc0+(f32((val8*val50)))+(f32((val5*val44)))+(f32((val2*val39)))+(f32((val7*val49)))+(f32((val4*val36)))+(f32((val1*val38)))+(f32((val6*val48)))+(f32((val3*val43)))+(f32((val0*val37))));
    acc1 = (acc1+(f32((val8*val51)))+(f32((val5*val45)))+(f32((val2*val40)))+(f32((val7*val50)))+(f32((val4*val44)))+(f32((val1*val39)))+(f32((val6*val49)))+(f32((val3*val36)))+(f32((val0*val38))));
    acc2 = (acc2+(f32((val8*val52)))+(f32((val5*val46)))+(f32((val2*val41)))+(f32((val7*val51)))+(f32((val4*val45)))+(f32((val1*val40)))+(f32((val6*val50)))+(f32((val3*val44)))+(f32((val0*val39))));
    acc3 = (acc3+(f32((val8*val53)))+(f32((val5*val47)))+(f32((val2*val42)))+(f32((val7*val52)))+(f32((val4*val46)))+(f32((val1*val41)))+(f32((val6*val51)))+(f32((val3*val45)))+(f32((val0*val40))));
    acc4 = (acc4+(f32((val17*val50)))+(f32((val14*val44)))+(f32((val11*val39)))+(f32((val16*val49)))+(f32((val13*val36)))+(f32((val10*val38)))+(f32((val15*val48)))+(f32((val9*val37)))+(f32((val12*val43))));
    acc5 = (acc5+(f32((val17*val51)))+(f32((val14*val45)))+(f32((val11*val40)))+(f32((val16*val50)))+(f32((val13*val44)))+(f32((val10*val39)))+(f32((val15*val49)))+(f32((val9*val38)))+(f32((val12*val36))));
    acc6 = (acc6+(f32((val17*val52)))+(f32((val14*val46)))+(f32((val11*val41)))+(f32((val16*val51)))+(f32((val13*val45)))+(f32((val10*val40)))+(f32((val15*val50)))+(f32((val9*val39)))+(f32((val12*val44))));
    acc7 = (acc7+(f32((val17*val53)))+(f32((val14*val47)))+(f32((val11*val42)))+(f32((val16*val52)))+(f32((val13*val46)))+(f32((val10*val41)))+(f32((val15*val51)))+(f32((val9*val40)))+(f32((val12*val45))));
    acc8 = (acc8+(f32((val26*val50)))+(f32((val23*val44)))+(f32((val20*val39)))+(f32((val25*val49)))+(f32((val22*val36)))+(f32((val19*val38)))+(f32((val24*val48)))+(f32((val18*val37)))+(f32((val21*val43))));
    acc9 = (acc9+(f32((val26*val51)))+(f32((val23*val45)))+(f32((val20*val40)))+(f32((val25*val50)))+(f32((val22*val44)))+(f32((val19*val39)))+(f32((val24*val49)))+(f32((val18*val38)))+(f32((val21*val36))));
    acc10 = (acc10+(f32((val26*val52)))+(f32((val23*val46)))+(f32((val20*val41)))+(f32((val25*val51)))+(f32((val22*val45)))+(f32((val19*val40)))+(f32((val24*val50)))+(f32((val18*val39)))+(f32((val21*val44))));
    acc11 = (acc11+(f32((val26*val53)))+(f32((val23*val47)))+(f32((val20*val42)))+(f32((val25*val52)))+(f32((val22*val46)))+(f32((val19*val41)))+(f32((val24*val51)))+(f32((val18*val40)))+(f32((val21*val45))));
    acc12 = (acc12+(f32((val35*val50)))+(f32((val32*val44)))+(f32((val29*val39)))+(f32((val34*val49)))+(f32((val31*val36)))+(f32((val28*val38)))+(f32((val33*val48)))+(f32((val27*val37)))+(f32((val30*val43))));
    acc13 = (acc13+(f32((val35*val51)))+(f32((val32*val45)))+(f32((val29*val40)))+(f32((val34*val50)))+(f32((val31*val44)))+(f32((val28*val39)))+(f32((val33*val49)))+(f32((val27*val38)))+(f32((val30*val36))));
    acc14 = (acc14+(f32((val35*val52)))+(f32((val32*val46)))+(f32((val29*val41)))+(f32((val34*val51)))+(f32((val31*val45)))+(f32((val28*val40)))+(f32((val33*val50)))+(f32((val27*val39)))+(f32((val30*val44))));
    acc15 = (acc15+(f32((val35*val53)))+(f32((val32*val47)))+(f32((val29*val42)))+(f32((val34*val52)))+(f32((val31*val46)))+(f32((val28*val41)))+(f32((val33*val51)))+(f32((val27*val40)))+(f32((val30*val45))));
  }
  var alu27 = ((gidx1<<14)+alu0+alu1+alu2+alu3);
  data0[alu27] = acc0;
  data0[(alu27+1)] = acc1;
  data0[(alu27+2)] = acc2;
  data0[(alu27+3)] = acc3;
  data0[(alu27+4096)] = acc4;
  data0[(alu27+4097)] = acc5;
  data0[(alu27+4098)] = acc6;
  data0[(alu27+4099)] = acc7;
  data0[(alu27+8192)] = acc8;
  data0[(alu27+8193)] = acc9;
  data0[(alu27+8194)] = acc10;
  data0[(alu27+8195)] = acc11;
  data0[(alu27+12288)] = acc12;
  data0[(alu27+12289)] = acc13;
  data0[(alu27+12290)] = acc14;
  data0[(alu27+12291)] = acc15;
}`;

const r_2_10_64_8_16_240_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 10 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 240; ridx0++) {
    var alu2 = ((gidx1*30720)+(lidx0*3840)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+960)];
    var val5 = data2[(alu2+961)];
    var val6 = data2[(alu2+962)];
    var val7 = data2[(alu2+963)];
    var val8 = data2[(alu2+1920)];
    var val9 = data2[(alu2+1921)];
    var val10 = data2[(alu2+1922)];
    var val11 = data2[(alu2+1923)];
    var val12 = data2[(alu2+2880)];
    var val13 = data2[(alu2+2881)];
    var val14 = data2[(alu2+2882)];
    var val15 = data2[(alu2+2883)];
    var alu3 = (alu0+(gidx2*3932160)+alu1+(ridx0<<14));
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
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<17)+(gidx2*1310720)+alu0+(lidx0<<14)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+4096);
  var val44 = data4[alu29];
  var alu30 = (alu25+4097);
  var val45 = data4[alu30];
  var alu31 = (alu25+4098);
  var val46 = data4[alu31];
  var alu32 = (alu25+4099);
  var val47 = data4[alu32];
  var alu33 = (alu25+8192);
  var val48 = data4[alu33];
  var alu34 = (alu25+8193);
  var val49 = data4[alu34];
  var alu35 = (alu25+8194);
  var val50 = data4[alu35];
  var alu36 = (alu25+8195);
  var val51 = data4[alu36];
  var alu37 = (alu25+12288);
  var val52 = data4[alu37];
  var alu38 = (alu25+12289);
  var val53 = data4[alu38];
  var alu39 = (alu25+12290);
  var val54 = data4[alu39];
  var alu40 = (alu25+12291);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const E_2_80_64_8_16_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<40);
  var alu1 = (alu0!=true);
  var alu2 = (gidx0<<6);
  var alu3 = (gidx1<<15);
  var alu4 = (lidx0<<12);
  var alu5 = (lidx1<<2);
  var alu6 = (alu3+(gidx2*1310720)+alu2+alu4+alu5);
  var val0 = select((f16(0.0f)), data1[alu6], alu0);
  var val1 = select((f16(0.0f)), data2[(alu6+-1310720)], alu1);
  var val2 = select((f16(0.0f)), data2[(alu6+-1310719)], alu1);
  var val3 = select((f16(0.0f)), data2[(alu6+-1310718)], alu1);
  var val4 = select((f16(0.0f)), data2[(alu6+-1310717)], alu1);
  var val5 = select((f16(0.0f)), data1[(alu6+1)], alu0);
  var val6 = select((f16(0.0f)), data1[(alu6+2)], alu0);
  var val7 = select((f16(0.0f)), data1[(alu6+3)], alu0);
  var alu7 = (alu3+(gidx2*2621440)+alu2+alu4+alu5);
  data0[alu7] = (val0+val1);
  data0[(alu7+1)] = (val5+val2);
  data0[(alu7+2)] = (val6+val3);
  data0[(alu7+3)] = (val7+val4);
}`;

const r_512_32_80_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu0 = ((gidx0*10240)+(lidx0*320)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    acc0 = (acc0+(f32(val3))+(f32(val2))+(f32(val1))+(f32(val0)));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_64_16_16n6 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = (f16((acc1*1.220703143189894e-05f)));
  }
}`;

const r_8_4_8_16_80_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[(lidx0+(gidx1<<3))];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 80; ridx0++) {
    var alu0 = ((gidx0*20480)+(gidx1*655360)+(lidx0*81920)+(lidx1*1280)+(ridx0<<2));
    var val1 = data1[alu0];
    var val2 = data1[(alu0+1)];
    var val3 = data1[(alu0+2)];
    var val4 = data1[(alu0+3)];
    var val5 = data1[(alu0+320)];
    var val6 = data1[(alu0+321)];
    var val7 = data1[(alu0+322)];
    var val8 = data1[(alu0+323)];
    var val9 = data1[(alu0+640)];
    var val10 = data1[(alu0+641)];
    var val11 = data1[(alu0+642)];
    var val12 = data1[(alu0+643)];
    var val13 = data1[(alu0+960)];
    var val14 = data1[(alu0+961)];
    var val15 = data1[(alu0+962)];
    var val16 = data1[(alu0+963)];
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
    acc0 = (acc0+(f32((alu3*alu3)))+(f32((alu2*alu2)))+(f32((alu1*alu1)))+(f32((alu16*alu16))));
    acc1 = (acc1+(f32((alu7*alu7)))+(f32((alu6*alu6)))+(f32((alu4*alu4)))+(f32((alu5*alu5))));
    acc2 = (acc2+(f32((alu11*alu11)))+(f32((alu10*alu10)))+(f32((alu8*alu8)))+(f32((alu9*alu9))));
    acc3 = (acc3+(f32((alu15*alu15)))+(f32((alu14*alu14)))+(f32((alu12*alu12)))+(f32((alu13*alu13))));
  }
  var alu22 = ((gidx0<<6)+(gidx1<<11)+(lidx0<<8)+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc1;
  data0[(alu22+2)] = acc2;
  data0[(alu22+3)] = acc3;
}`;

const r_64_16_16n7 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
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
    data0[gidx0] = sqrt((1/((f16((acc1*1.220703143189894e-05f)))+(f16(1e-05f)))));
  }
}`;

const E_2_80_64_8_16_4n1 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data4[alu0];
  var val1 = data5[alu0];
  var alu1 = ((gidx2<<5)+(alu0/20));
  var val2 = data2[alu1];
  var val3 = data3[alu1];
  var alu2 = ((gidx1<<15)+(gidx2*2621440)+(gidx0<<6)+(lidx0<<12)+(lidx1<<2));
  var val4 = data1[alu2];
  var alu3 = (alu2+1);
  var val5 = data1[alu3];
  var alu4 = (alu2+2);
  var val6 = data1[alu4];
  var alu5 = (alu2+3);
  var val7 = data1[alu5];
  var alu6 = -val1;
  var alu7 = (val0*val3*(val5-val2));
  data0[alu3] = ((1/(exp2(((alu6-alu7)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu7));
  var alu9 = (val0*val3*(val6-val2));
  data0[alu4] = ((1/(exp2(((alu6-alu9)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu9));
  var alu11 = (val0*val3*(val7-val2));
  data0[alu5] = ((1/(exp2(((alu6-alu11)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu11));
  var alu13 = (val0*val3*(val4-val2));
  data0[alu2] = ((1/(exp2(((alu6-alu13)*(f16(1.4426950408889634f))))+(f16(1.0f))))*(val1+alu13));
}`;

const r_2_80_8_8_16_640_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 80 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 640; ridx0++) {
    var alu7 = ((gidx1*23040)+(ridx0*9));
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+5760)];
    var val10 = data2[(alu7+5761)];
    var val11 = data2[(alu7+5762)];
    var val12 = data2[(alu7+5763)];
    var val13 = data2[(alu7+5764)];
    var val14 = data2[(alu7+5765)];
    var val15 = data2[(alu7+5766)];
    var val16 = data2[(alu7+5767)];
    var val17 = data2[(alu7+5768)];
    var val18 = data2[(alu7+11520)];
    var val19 = data2[(alu7+11521)];
    var val20 = data2[(alu7+11522)];
    var val21 = data2[(alu7+11523)];
    var val22 = data2[(alu7+11524)];
    var val23 = data2[(alu7+11525)];
    var val24 = data2[(alu7+11526)];
    var val25 = data2[(alu7+11527)];
    var val26 = data2[(alu7+11528)];
    var val27 = data2[(alu7+17280)];
    var val28 = data2[(alu7+17281)];
    var val29 = data2[(alu7+17282)];
    var val30 = data2[(alu7+17283)];
    var val31 = data2[(alu7+17284)];
    var val32 = data2[(alu7+17285)];
    var val33 = data2[(alu7+17286)];
    var val34 = data2[(alu7+17287)];
    var val35 = data2[(alu7+17288)];
    var alu8 = (alu0+alu1+(gidx2*2621440)+(ridx0<<12)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-65)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-64)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-63)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-62)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-61)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-60)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+63)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+64)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+65)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+66)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+67)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var alu26 = (gidx1<<2);
  var val54 = data3[alu26];
  var val55 = data4[alu26];
  var alu27 = (alu26+1);
  var val56 = data3[alu27];
  var val57 = data4[alu27];
  var alu28 = (alu26+2);
  var val58 = data3[alu28];
  var val59 = data4[alu28];
  var alu29 = (alu26+3);
  var val60 = data3[alu29];
  var val61 = data4[alu29];
  var alu30 = ((gidx1<<14)+(gidx2*1310720)+alu0+alu1+alu2);
  data0[alu30] = (val55+(f16(acc0))+val54);
  data0[(alu30+1)] = (val55+(f16(acc4))+val54);
  data0[(alu30+2)] = (val55+(f16(acc8))+val54);
  data0[(alu30+3)] = (val55+(f16(acc12))+val54);
  data0[(alu30+4096)] = (val57+(f16(acc1))+val56);
  data0[(alu30+4097)] = (val57+(f16(acc5))+val56);
  data0[(alu30+4098)] = (val57+(f16(acc9))+val56);
  data0[(alu30+4099)] = (val57+(f16(acc13))+val56);
  data0[(alu30+8192)] = (val59+(f16(acc2))+val58);
  data0[(alu30+8193)] = (val59+(f16(acc6))+val58);
  data0[(alu30+8194)] = (val59+(f16(acc10))+val58);
  data0[(alu30+8195)] = (val59+(f16(acc14))+val58);
  data0[(alu30+12288)] = (val61+(f16(acc3))+val60);
  data0[(alu30+12289)] = (val61+(f16(acc7))+val60);
  data0[(alu30+12290)] = (val61+(f16(acc11))+val60);
  data0[(alu30+12291)] = (val61+(f16(acc15))+val60);
}`;

const r_2_10_64_8_16_160_4_4_4 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@group(0) @binding(5)var<storage,read_write>data4:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 10 */
  var gidx2 = i32(gindex.z); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 160; ridx0++) {
    var alu2 = ((gidx1*20480)+(lidx0*2560)+(ridx0<<2));
    var val0 = data2[alu2];
    var val1 = data2[(alu2+1)];
    var val2 = data2[(alu2+2)];
    var val3 = data2[(alu2+3)];
    var val4 = data2[(alu2+640)];
    var val5 = data2[(alu2+641)];
    var val6 = data2[(alu2+642)];
    var val7 = data2[(alu2+643)];
    var val8 = data2[(alu2+1280)];
    var val9 = data2[(alu2+1281)];
    var val10 = data2[(alu2+1282)];
    var val11 = data2[(alu2+1283)];
    var val12 = data2[(alu2+1920)];
    var val13 = data2[(alu2+1921)];
    var val14 = data2[(alu2+1922)];
    var val15 = data2[(alu2+1923)];
    var alu3 = (alu0+(gidx2*2621440)+alu1+(ridx0<<14));
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
    acc0 = (acc0+(f32((val28*val3)))+(f32((val24*val2)))+(f32((val20*val1)))+(f32((val16*val0))));
    acc1 = (acc1+(f32((val28*val7)))+(f32((val24*val6)))+(f32((val20*val5)))+(f32((val16*val4))));
    acc2 = (acc2+(f32((val28*val11)))+(f32((val24*val10)))+(f32((val20*val9)))+(f32((val16*val8))));
    acc3 = (acc3+(f32((val28*val15)))+(f32((val24*val14)))+(f32((val20*val13)))+(f32((val16*val12))));
    acc4 = (acc4+(f32((val29*val3)))+(f32((val25*val2)))+(f32((val17*val0)))+(f32((val21*val1))));
    acc5 = (acc5+(f32((val29*val7)))+(f32((val25*val6)))+(f32((val17*val4)))+(f32((val21*val5))));
    acc6 = (acc6+(f32((val29*val11)))+(f32((val25*val10)))+(f32((val17*val8)))+(f32((val21*val9))));
    acc7 = (acc7+(f32((val29*val15)))+(f32((val25*val14)))+(f32((val17*val12)))+(f32((val21*val13))));
    acc8 = (acc8+(f32((val30*val3)))+(f32((val26*val2)))+(f32((val18*val0)))+(f32((val22*val1))));
    acc9 = (acc9+(f32((val30*val7)))+(f32((val26*val6)))+(f32((val18*val4)))+(f32((val22*val5))));
    acc10 = (acc10+(f32((val30*val11)))+(f32((val26*val10)))+(f32((val18*val8)))+(f32((val22*val9))));
    acc11 = (acc11+(f32((val30*val15)))+(f32((val26*val14)))+(f32((val18*val12)))+(f32((val22*val13))));
    acc12 = (acc12+(f32((val31*val3)))+(f32((val27*val2)))+(f32((val19*val0)))+(f32((val23*val1))));
    acc13 = (acc13+(f32((val31*val7)))+(f32((val27*val6)))+(f32((val19*val4)))+(f32((val23*val5))));
    acc14 = (acc14+(f32((val31*val11)))+(f32((val27*val10)))+(f32((val19*val8)))+(f32((val23*val9))));
    acc15 = (acc15+(f32((val31*val15)))+(f32((val27*val14)))+(f32((val19*val12)))+(f32((val23*val13))));
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
  var alu25 = ((gidx1<<17)+(gidx2*1310720)+alu0+(lidx0<<14)+alu1);
  var val40 = data4[alu25];
  var alu26 = (alu25+1);
  var val41 = data4[alu26];
  var alu27 = (alu25+2);
  var val42 = data4[alu27];
  var alu28 = (alu25+3);
  var val43 = data4[alu28];
  var alu29 = (alu25+4096);
  var val44 = data4[alu29];
  var alu30 = (alu25+4097);
  var val45 = data4[alu30];
  var alu31 = (alu25+4098);
  var val46 = data4[alu31];
  var alu32 = (alu25+4099);
  var val47 = data4[alu32];
  var alu33 = (alu25+8192);
  var val48 = data4[alu33];
  var alu34 = (alu25+8193);
  var val49 = data4[alu34];
  var alu35 = (alu25+8194);
  var val50 = data4[alu35];
  var alu36 = (alu25+8195);
  var val51 = data4[alu36];
  var alu37 = (alu25+12288);
  var val52 = data4[alu37];
  var alu38 = (alu25+12289);
  var val53 = data4[alu38];
  var alu39 = (alu25+12290);
  var val54 = data4[alu39];
  var alu40 = (alu25+12291);
  var val55 = data4[alu40];
  data0[alu26] = ((f16(val41))+val33+(f16(acc4))+val32);
  data0[alu27] = ((f16(val42))+val33+(f16(acc8))+val32);
  data0[alu28] = ((f16(val43))+val33+(f16(acc12))+val32);
  data0[alu29] = ((f16(val44))+val35+(f16(acc1))+val34);
  data0[alu30] = ((f16(val45))+val35+(f16(acc5))+val34);
  data0[alu31] = ((f16(val46))+val35+(f16(acc9))+val34);
  data0[alu32] = ((f16(val47))+val35+(f16(acc13))+val34);
  data0[alu33] = ((f16(val48))+val37+(f16(acc2))+val36);
  data0[alu34] = ((f16(val49))+val37+(f16(acc6))+val36);
  data0[alu35] = ((f16(val50))+val37+(f16(acc10))+val36);
  data0[alu36] = ((f16(val51))+val37+(f16(acc14))+val36);
  data0[alu37] = ((f16(val52))+val39+(f16(acc3))+val38);
  data0[alu38] = ((f16(val53))+val39+(f16(acc7))+val38);
  data0[alu39] = ((f16(val54))+val39+(f16(acc11))+val38);
  data0[alu40] = ((f16(val55))+val39+(f16(acc15))+val38);
  data0[alu25] = ((f16(val40))+val33+(f16(acc0))+val32);
}`;

const r_2_8_8_16_320_4_4_3_3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f16>;
@group(0) @binding(2)var<storage,read_write>data1:array<f16>;
@group(0) @binding(3)var<storage,read_write>data2:array<f16>;
@group(0) @binding(4)var<storage,read_write>data3:array<f16>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 2 */
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
  for (var ridx0 = 0; ridx0 < 320; ridx0++) {
    var alu7 = (ridx0*9);
    var val0 = data2[alu7];
    var val1 = data2[(alu7+1)];
    var val2 = data2[(alu7+2)];
    var val3 = data2[(alu7+3)];
    var val4 = data2[(alu7+4)];
    var val5 = data2[(alu7+5)];
    var val6 = data2[(alu7+6)];
    var val7 = data2[(alu7+7)];
    var val8 = data2[(alu7+8)];
    var val9 = data2[(alu7+2880)];
    var val10 = data2[(alu7+2881)];
    var val11 = data2[(alu7+2882)];
    var val12 = data2[(alu7+2883)];
    var val13 = data2[(alu7+2884)];
    var val14 = data2[(alu7+2885)];
    var val15 = data2[(alu7+2886)];
    var val16 = data2[(alu7+2887)];
    var val17 = data2[(alu7+2888)];
    var val18 = data2[(alu7+5760)];
    var val19 = data2[(alu7+5761)];
    var val20 = data2[(alu7+5762)];
    var val21 = data2[(alu7+5763)];
    var val22 = data2[(alu7+5764)];
    var val23 = data2[(alu7+5765)];
    var val24 = data2[(alu7+5766)];
    var val25 = data2[(alu7+5767)];
    var val26 = data2[(alu7+5768)];
    var val27 = data2[(alu7+8640)];
    var val28 = data2[(alu7+8641)];
    var val29 = data2[(alu7+8642)];
    var val30 = data2[(alu7+8643)];
    var val31 = data2[(alu7+8644)];
    var val32 = data2[(alu7+8645)];
    var val33 = data2[(alu7+8646)];
    var val34 = data2[(alu7+8647)];
    var val35 = data2[(alu7+8648)];
    var alu8 = (alu0+alu1+(gidx1*1310720)+(ridx0<<12)+alu2);
    var val36 = data1[alu8];
    var val37 = select((f16(0.0f)), data1[(alu8+-65)], (alu3&alu4));
    var val38 = select((f16(0.0f)), data1[(alu8+-64)], alu4);
    var val39 = select((f16(0.0f)), data1[(alu8+-63)], alu4);
    var val40 = select((f16(0.0f)), data1[(alu8+-62)], alu4);
    var val41 = select((f16(0.0f)), data1[(alu8+-61)], alu4);
    var val42 = select((f16(0.0f)), data1[(alu8+-60)], (alu6&alu4));
    var val43 = select((f16(0.0f)), data1[(alu8+-1)], alu3);
    var val44 = data1[(alu8+1)];
    var val45 = data1[(alu8+2)];
    var val46 = data1[(alu8+3)];
    var val47 = select((f16(0.0f)), data1[(alu8+4)], alu6);
    var val48 = select((f16(0.0f)), data1[(alu8+63)], (alu5&alu3));
    var val49 = select((f16(0.0f)), data1[(alu8+64)], alu5);
    var val50 = select((f16(0.0f)), data1[(alu8+65)], alu5);
    var val51 = select((f16(0.0f)), data1[(alu8+66)], alu5);
    var val52 = select((f16(0.0f)), data1[(alu8+67)], alu5);
    var val53 = select((f16(0.0f)), data1[(alu8+68)], (alu6&alu5));
    acc0 = (acc0+(f32((val50*val8)))+(f32((val44*val5)))+(f32((val39*val2)))+(f32((val49*val7)))+(f32((val36*val4)))+(f32((val38*val1)))+(f32((val48*val6)))+(f32((val37*val0)))+(f32((val43*val3))));
    acc1 = (acc1+(f32((val50*val17)))+(f32((val44*val14)))+(f32((val39*val11)))+(f32((val49*val16)))+(f32((val36*val13)))+(f32((val38*val10)))+(f32((val48*val15)))+(f32((val37*val9)))+(f32((val43*val12))));
    acc2 = (acc2+(f32((val50*val26)))+(f32((val44*val23)))+(f32((val39*val20)))+(f32((val49*val25)))+(f32((val36*val22)))+(f32((val38*val19)))+(f32((val48*val24)))+(f32((val37*val18)))+(f32((val43*val21))));
    acc3 = (acc3+(f32((val50*val35)))+(f32((val44*val32)))+(f32((val39*val29)))+(f32((val49*val34)))+(f32((val36*val31)))+(f32((val38*val28)))+(f32((val48*val33)))+(f32((val37*val27)))+(f32((val43*val30))));
    acc4 = (acc4+(f32((val51*val8)))+(f32((val45*val5)))+(f32((val40*val2)))+(f32((val50*val7)))+(f32((val44*val4)))+(f32((val39*val1)))+(f32((val49*val6)))+(f32((val38*val0)))+(f32((val36*val3))));
    acc5 = (acc5+(f32((val51*val17)))+(f32((val45*val14)))+(f32((val40*val11)))+(f32((val50*val16)))+(f32((val44*val13)))+(f32((val39*val10)))+(f32((val49*val15)))+(f32((val38*val9)))+(f32((val36*val12))));
    acc6 = (acc6+(f32((val51*val26)))+(f32((val45*val23)))+(f32((val40*val20)))+(f32((val50*val25)))+(f32((val44*val22)))+(f32((val39*val19)))+(f32((val49*val24)))+(f32((val38*val18)))+(f32((val36*val21))));
    acc7 = (acc7+(f32((val51*val35)))+(f32((val45*val32)))+(f32((val40*val29)))+(f32((val50*val34)))+(f32((val44*val31)))+(f32((val39*val28)))+(f32((val49*val33)))+(f32((val38*val27)))+(f32((val36*val30))));
    acc8 = (acc8+(f32((val52*val8)))+(f32((val46*val5)))+(f32((val41*val2)))+(f32((val51*val7)))+(f32((val45*val4)))+(f32((val40*val1)))+(f32((val50*val6)))+(f32((val39*val0)))+(f32((val44*val3))));
    acc9 = (acc9+(f32((val52*val17)))+(f32((val46*val14)))+(f32((val41*val11)))+(f32((val51*val16)))+(f32((val45*val13)))+(f32((val40*val10)))+(f32((val50*val15)))+(f32((val39*val9)))+(f32((val44*val12))));
    acc10 = (acc10+(f32((val52*val26)))+(f32((val46*val23)))+(f32((val41*val20)))+(f32((val51*val25)))+(f32((val45*val22)))+(f32((val40*val19)))+(f32((val50*val24)))+(f32((val39*val18)))+(f32((val44*val21))));
    acc11 = (acc11+(f32((val52*val35)))+(f32((val46*val32)))+(f32((val41*val29)))+(f32((val51*val34)))+(f32((val45*val31)))+(f32((val40*val28)))+(f32((val50*val33)))+(f32((val39*val27)))+(f32((val44*val30))));
    acc12 = (acc12+(f32((val53*val8)))+(f32((val47*val5)))+(f32((val42*val2)))+(f32((val52*val7)))+(f32((val46*val4)))+(f32((val41*val1)))+(f32((val51*val6)))+(f32((val40*val0)))+(f32((val45*val3))));
    acc13 = (acc13+(f32((val53*val17)))+(f32((val47*val14)))+(f32((val42*val11)))+(f32((val52*val16)))+(f32((val46*val13)))+(f32((val41*val10)))+(f32((val51*val15)))+(f32((val40*val9)))+(f32((val45*val12))));
    acc14 = (acc14+(f32((val53*val26)))+(f32((val47*val23)))+(f32((val42*val20)))+(f32((val52*val25)))+(f32((val46*val22)))+(f32((val41*val19)))+(f32((val51*val24)))+(f32((val40*val18)))+(f32((val45*val21))));
    acc15 = (acc15+(f32((val53*val35)))+(f32((val47*val32)))+(f32((val42*val29)))+(f32((val52*val34)))+(f32((val46*val31)))+(f32((val41*val28)))+(f32((val51*val33)))+(f32((val40*val27)))+(f32((val45*val30))));
  }
  var val54 = data3[0];
  var val55 = data3[1];
  var val56 = data3[2];
  var val57 = data3[3];
  var alu26 = (alu0+(gidx1<<14)+alu1+alu2);
  data0[alu26] = ((f16(acc0))+val54);
  data0[(alu26+1)] = ((f16(acc4))+val54);
  data0[(alu26+2)] = ((f16(acc8))+val54);
  data0[(alu26+3)] = ((f16(acc12))+val54);
  data0[(alu26+4096)] = ((f16(acc1))+val55);
  data0[(alu26+4097)] = ((f16(acc5))+val55);
  data0[(alu26+4098)] = ((f16(acc9))+val55);
  data0[(alu26+4099)] = ((f16(acc13))+val55);
  data0[(alu26+8192)] = ((f16(acc2))+val56);
  data0[(alu26+8193)] = ((f16(acc6))+val56);
  data0[(alu26+8194)] = ((f16(acc10))+val56);
  data0[(alu26+8195)] = ((f16(acc14))+val56);
  data0[(alu26+12288)] = ((f16(acc3))+val57);
  data0[(alu26+12289)] = ((f16(acc7))+val57);
  data0[(alu26+12290)] = ((f16(acc11))+val57);
  data0[(alu26+12291)] = ((f16(acc15))+val57);
}`;

const E_128_32_4n3 = `enable f16;
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4:array<f16>;
@group(0) @binding(6)var<storage,read_write>data5:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6:array<f32>;
@group(0) @binding(8)var<storage,read_write>data7:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 32 */
  var val0 = data1[0];
  var val1 = data3[0];
  var val2 = data5[0];
  var val3 = data6[0];
  var val4 = data7[0];
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val5 = data4[alu0];
  var val6 = data2[alu0];
  var alu1 = (alu0+1);
  var val7 = data4[alu1];
  var val8 = data2[alu1];
  var alu2 = (alu0+2);
  var val9 = data4[alu2];
  var val10 = data2[alu2];
  var alu3 = (alu0+3);
  var val11 = data4[alu3];
  var val12 = data2[alu3];
  var val13 = data4[(alu0+16384)];
  var val14 = data4[(alu0+16385)];
  var val15 = data4[(alu0+16386)];
  var val16 = data4[(alu0+16387)];
  var alu4 = (1/val3);
  var alu5 = ((f32(val5))+((f32((val13-val5)))*val2));
  var alu6 = ((f32(val7))+((f32((val14-val7)))*val2));
  var alu7 = ((f32(val9))+((f32((val15-val9)))*val2));
  var alu8 = ((f32(val11))+((f32((val16-val11)))*val2));
  data0[alu1] = ((val0*alu4*(val8-(val1*alu6)))+(val4*alu6));
  data0[alu2] = ((val0*alu4*(val10-(val1*alu7)))+(val4*alu7));
  data0[alu3] = ((val0*alu4*(val12-(val1*alu8)))+(val4*alu8));
  data0[alu0] = ((val0*alu4*(val6-(val1*alu5)))+(val4*alu5));
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);


    const buf_0 = createEmptyBuf(device, 4);;
    const input5 = createEmptyBuf(device, 4);;
    const buf_1 = createEmptyBuf(device, 4);;
    const input4 = createEmptyBuf(device, 4);;
    const buf_2 = createEmptyBuf(device, 32768);;
    const input2 = createEmptyBuf(device, 65536);;
    const buf_3 = createEmptyBuf(device, 640);;
    const input3 = createEmptyBuf(device, 4);;
    const buf_4 = createEmptyBuf(device, 640);;
    const buf_5 = createEmptyBuf(device, 236544);;
    const input0 = createEmptyBuf(device, 236544);;
    const input1 = createEmptyBuf(device, 236544);;
    const buf_6 = createEmptyBuf(device, 4);;
    const buf_7 = createEmptyBuf(device, 4);;
    const buf_8 = createEmptyBuf(device, 5242880);;
    const buf_9 = createWeightBuf(device, 23040, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.0.0.weight']));
    const buf_10 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.0.0.bias']));
    const buf_11 = createEmptyBuf(device, 640);;
    const buf_12 = createEmptyBuf(device, 98560);;
    const buf_13 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_14 = createEmptyBuf(device, 98560);;
    const buf_15 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_16 = createEmptyBuf(device, 98560);;
    const buf_17 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_18 = createEmptyBuf(device, 98560);;
    const buf_19 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_20 = createEmptyBuf(device, 197120);;
    const buf_21 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_22 = createEmptyBuf(device, 197120);;
    const buf_23 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_24 = createEmptyBuf(device, 197120);;
    const buf_25 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_26 = createEmptyBuf(device, 197120);;
    const buf_27 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_28 = createEmptyBuf(device, 394240);;
    const buf_29 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_30 = createEmptyBuf(device, 394240);;
    const buf_31 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_32 = createEmptyBuf(device, 394240);;
    const buf_33 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_34 = createEmptyBuf(device, 394240);;
    const buf_35 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_36 = createEmptyBuf(device, 394240);;
    const buf_37 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_38 = createEmptyBuf(device, 394240);;
    const buf_39 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_40 = createEmptyBuf(device, 394240);;
    const buf_41 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_42 = createEmptyBuf(device, 394240);;
    const buf_43 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_44 = createEmptyBuf(device, 394240);;
    const buf_45 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_46 = createEmptyBuf(device, 394240);;
    const buf_47 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_48 = createEmptyBuf(device, 394240);;
    const buf_49 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_50 = createEmptyBuf(device, 394240);;
    const buf_51 = createWeightBuf(device, 1966080, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_52 = createEmptyBuf(device, 197120);;
    const buf_53 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_54 = createEmptyBuf(device, 197120);;
    const buf_55 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_56 = createEmptyBuf(device, 197120);;
    const buf_57 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_58 = createEmptyBuf(device, 197120);;
    const buf_59 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_60 = createEmptyBuf(device, 197120);;
    const buf_61 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_62 = createEmptyBuf(device, 197120);;
    const buf_63 = createWeightBuf(device, 983040, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_64 = createEmptyBuf(device, 98560);;
    const buf_65 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_66 = createEmptyBuf(device, 98560);;
    const buf_67 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_68 = createEmptyBuf(device, 98560);;
    const buf_69 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_70 = createEmptyBuf(device, 98560);;
    const buf_71 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_72 = createEmptyBuf(device, 98560);;
    const buf_73 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight']));
    const buf_74 = createEmptyBuf(device, 98560);;
    const buf_75 = createWeightBuf(device, 491520, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight']));
    const buf_76 = createEmptyBuf(device, 65536);;
    const buf_77 = createEmptyBuf(device, 2560);;
    const buf_78 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.time_embed.0.weight']));
    const buf_79 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.time_embed.0.bias']));
    const buf_80 = createEmptyBuf(device, 128);;
    const buf_81 = createEmptyBuf(device, 2560);;
    const buf_82 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.time_embed.2.weight']));
    const buf_83 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.time_embed.2.bias']));
    const buf_84 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.emb_layers.1.weight']));
    const buf_85 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.emb_layers.1.bias']));
    const buf_86 = createEmptyBuf(device, 640);;
    const buf_87 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.emb_layers.1.weight']));
    const buf_88 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.emb_layers.1.bias']));
    const buf_89 = createEmptyBuf(device, 1280);;
    const buf_90 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.emb_layers.1.weight']));
    const buf_91 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.emb_layers.1.bias']));
    const buf_92 = createEmptyBuf(device, 1280);;
    const buf_93 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.emb_layers.1.weight']));
    const buf_94 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.emb_layers.1.bias']));
    const buf_95 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.emb_layers.1.weight']));
    const buf_96 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.emb_layers.1.bias']));
    const buf_97 = createEmptyBuf(device, 2560);;
    const buf_98 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.emb_layers.1.weight']));
    const buf_99 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.emb_layers.1.bias']));
    const buf_100 = createEmptyBuf(device, 2560);;
    const buf_101 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.emb_layers.1.weight']));
    const buf_102 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.emb_layers.1.bias']));
    const buf_103 = createEmptyBuf(device, 2560);;
    const buf_104 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.emb_layers.1.weight']));
    const buf_105 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.emb_layers.1.bias']));
    const buf_106 = createEmptyBuf(device, 2560);;
    const buf_107 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.emb_layers.1.weight']));
    const buf_108 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.emb_layers.1.bias']));
    const buf_109 = createEmptyBuf(device, 2560);;
    const buf_110 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.emb_layers.1.weight']));
    const buf_111 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.emb_layers.1.bias']));
    const buf_112 = createEmptyBuf(device, 2560);;
    const buf_113 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.emb_layers.1.weight']));
    const buf_114 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.emb_layers.1.bias']));
    const buf_115 = createEmptyBuf(device, 2560);;
    const buf_116 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.emb_layers.1.weight']));
    const buf_117 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.emb_layers.1.bias']));
    const buf_118 = createEmptyBuf(device, 2560);;
    const buf_119 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.emb_layers.1.weight']));
    const buf_120 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.emb_layers.1.bias']));
    const buf_121 = createEmptyBuf(device, 2560);;
    const buf_122 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.emb_layers.1.weight']));
    const buf_123 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.emb_layers.1.bias']));
    const buf_124 = createEmptyBuf(device, 2560);;
    const buf_125 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.emb_layers.1.weight']));
    const buf_126 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.emb_layers.1.bias']));
    const buf_127 = createEmptyBuf(device, 2560);;
    const buf_128 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.emb_layers.1.weight']));
    const buf_129 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.emb_layers.1.bias']));
    const buf_130 = createEmptyBuf(device, 1280);;
    const buf_131 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.emb_layers.1.weight']));
    const buf_132 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.emb_layers.1.bias']));
    const buf_133 = createEmptyBuf(device, 1280);;
    const buf_134 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.emb_layers.1.weight']));
    const buf_135 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.emb_layers.1.bias']));
    const buf_136 = createEmptyBuf(device, 1280);;
    const buf_137 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.emb_layers.1.weight']));
    const buf_138 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.emb_layers.1.bias']));
    const buf_139 = createEmptyBuf(device, 640);;
    const buf_140 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.emb_layers.1.weight']));
    const buf_141 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.emb_layers.1.bias']));
    const buf_142 = createEmptyBuf(device, 640);;
    const buf_143 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.emb_layers.1.weight']));
    const buf_144 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.emb_layers.1.bias']));
    const buf_145 = createEmptyBuf(device, 640);;
    const buf_146 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.emb_layers.1.weight']));
    const buf_147 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.emb_layers.1.bias']));
    const buf_148 = createEmptyBuf(device, 128);;
    const buf_149 = createEmptyBuf(device, 5242880);;
    const buf_150 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.in_layers.0.weight']));
    const buf_151 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.in_layers.0.bias']));
    const buf_152 = createEmptyBuf(device, 5242880);;
    const buf_153 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.in_layers.2.weight']));
    const buf_154 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.in_layers.2.bias']));
    const buf_155 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.out_layers.0.weight']));
    const buf_156 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.out_layers.0.bias']));
    const buf_157 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.out_layers.3.weight']));
    const buf_158 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.0.out_layers.3.bias']));
    const buf_159 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.norm.weight']));
    const buf_160 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.norm.bias']));
    const buf_161 = createEmptyBuf(device, 5242880);;
    const buf_162 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.proj_in.weight']));
    const buf_163 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.proj_in.bias']));
    const buf_164 = createEmptyBuf(device, 16384);;
    const buf_165 = createEmptyBuf(device, 16384);;
    const buf_166 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight']));
    const buf_167 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias']));
    const buf_168 = createEmptyBuf(device, 5242880);;
    const buf_169 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_170 = createEmptyBuf(device, 5242880);;
    const buf_171 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_172 = createEmptyBuf(device, 5242880);;
    const buf_173 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_174 = createEmptyBuf(device, 1073741824);;
    const buf_175 = createEmptyBuf(device, 262144);;
    const buf_176 = createEmptyBuf(device, 262144);;
    const buf_177 = createEmptyBuf(device, 536870912);;
    const buf_178 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_179 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_180 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight']));
    const buf_181 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias']));
    const buf_182 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_183 = createEmptyBuf(device, 20185088);;
    const buf_184 = createEmptyBuf(device, 10092544);;
    const buf_185 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_186 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_187 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight']));
    const buf_188 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias']));
    const buf_189 = createEmptyBuf(device, 41943040);;
    const buf_190 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_191 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_192 = createEmptyBuf(device, 20971520);;
    const buf_193 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_194 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_195 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.proj_out.weight']));
    const buf_196 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.1.1.proj_out.bias']));
    const buf_197 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.in_layers.0.weight']));
    const buf_198 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.in_layers.0.bias']));
    const buf_199 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.in_layers.2.weight']));
    const buf_200 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.in_layers.2.bias']));
    const buf_201 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.out_layers.0.weight']));
    const buf_202 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.out_layers.0.bias']));
    const buf_203 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.out_layers.3.weight']));
    const buf_204 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.0.out_layers.3.bias']));
    const buf_205 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.norm.weight']));
    const buf_206 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.norm.bias']));
    const buf_207 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.proj_in.weight']));
    const buf_208 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.proj_in.bias']));
    const buf_209 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.weight']));
    const buf_210 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.bias']));
    const buf_211 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_212 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_213 = createEmptyBuf(device, 5242880);;
    const buf_214 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_215 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_216 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_217 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.weight']));
    const buf_218 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.bias']));
    const buf_219 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_220 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_221 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_222 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.weight']));
    const buf_223 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.bias']));
    const buf_224 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_225 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_226 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_227 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_228 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.proj_out.weight']));
    const buf_229 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.2.1.proj_out.bias']));
    const buf_230 = createEmptyBuf(device, 1310720);;
    const buf_231 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.3.0.op.weight']));
    const buf_232 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.3.0.op.bias']));
    const buf_233 = createEmptyBuf(device, 1310720);;
    const buf_234 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.in_layers.0.weight']));
    const buf_235 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.in_layers.0.bias']));
    const buf_236 = createEmptyBuf(device, 2621440);;
    const buf_237 = createWeightBuf(device, 3686400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.in_layers.2.weight']));
    const buf_238 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.in_layers.2.bias']));
    const buf_239 = createEmptyBuf(device, 2621440);;
    const buf_240 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.out_layers.0.weight']));
    const buf_241 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.out_layers.0.bias']));
    const buf_242 = createEmptyBuf(device, 5242880);;
    const buf_243 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.out_layers.3.weight']));
    const buf_244 = createWeightBuf(device, 409600, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.skip_connection.weight']));
    const buf_245 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.skip_connection.bias']));
    const buf_246 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.0.out_layers.3.bias']));
    const buf_247 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.norm.weight']));
    const buf_248 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.norm.bias']));
    const buf_249 = createEmptyBuf(device, 2621440);;
    const buf_250 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.proj_in.weight']));
    const buf_251 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.proj_in.bias']));
    const buf_252 = createEmptyBuf(device, 4096);;
    const buf_253 = createEmptyBuf(device, 4096);;
    const buf_254 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight']));
    const buf_255 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias']));
    const buf_256 = createEmptyBuf(device, 2621440);;
    const buf_257 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_258 = createEmptyBuf(device, 2621440);;
    const buf_259 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_260 = createEmptyBuf(device, 2621440);;
    const buf_261 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_262 = createEmptyBuf(device, 67108864);;
    const buf_263 = createEmptyBuf(device, 65536);;
    const buf_264 = createEmptyBuf(device, 33554432);;
    const buf_265 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_266 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_267 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight']));
    const buf_268 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias']));
    const buf_269 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_270 = createEmptyBuf(device, 5046272);;
    const buf_271 = createEmptyBuf(device, 2523136);;
    const buf_272 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_273 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_274 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight']));
    const buf_275 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias']));
    const buf_276 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_277 = createWeightBuf(device, 10240, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_278 = createEmptyBuf(device, 10485760);;
    const buf_279 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_280 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_281 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.proj_out.weight']));
    const buf_282 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.4.1.proj_out.bias']));
    const buf_283 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.in_layers.0.weight']));
    const buf_284 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.in_layers.0.bias']));
    const buf_285 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.in_layers.2.weight']));
    const buf_286 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.in_layers.2.bias']));
    const buf_287 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.out_layers.0.weight']));
    const buf_288 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.out_layers.0.bias']));
    const buf_289 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.out_layers.3.weight']));
    const buf_290 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.0.out_layers.3.bias']));
    const buf_291 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.norm.weight']));
    const buf_292 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.norm.bias']));
    const buf_293 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.proj_in.weight']));
    const buf_294 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.proj_in.bias']));
    const buf_295 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight']));
    const buf_296 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias']));
    const buf_297 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_298 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_299 = createEmptyBuf(device, 2621440);;
    const buf_300 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_301 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_302 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_303 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight']));
    const buf_304 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias']));
    const buf_305 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_306 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_307 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_308 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight']));
    const buf_309 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias']));
    const buf_310 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_311 = createWeightBuf(device, 10240, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_312 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_313 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_314 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.proj_out.weight']));
    const buf_315 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.5.1.proj_out.bias']));
    const buf_316 = createEmptyBuf(device, 655360);;
    const buf_317 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.6.0.op.weight']));
    const buf_318 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.6.0.op.bias']));
    const buf_319 = createEmptyBuf(device, 655360);;
    const buf_320 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.in_layers.0.weight']));
    const buf_321 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.in_layers.0.bias']));
    const buf_322 = createWeightBuf(device, 14745600, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.in_layers.2.weight']));
    const buf_323 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.in_layers.2.bias']));
    const buf_324 = createEmptyBuf(device, 1310720);;
    const buf_325 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.out_layers.0.weight']));
    const buf_326 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.out_layers.0.bias']));
    const buf_327 = createEmptyBuf(device, 2621440);;
    const buf_328 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.out_layers.3.weight']));
    const buf_329 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.skip_connection.weight']));
    const buf_330 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.skip_connection.bias']));
    const buf_331 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.0.out_layers.3.bias']));
    const buf_332 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.norm.weight']));
    const buf_333 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.norm.bias']));
    const buf_334 = createEmptyBuf(device, 1310720);;
    const buf_335 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.proj_in.weight']));
    const buf_336 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.proj_in.bias']));
    const buf_337 = createEmptyBuf(device, 1024);;
    const buf_338 = createEmptyBuf(device, 1024);;
    const buf_339 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight']));
    const buf_340 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias']));
    const buf_341 = createEmptyBuf(device, 1310720);;
    const buf_342 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_343 = createEmptyBuf(device, 1310720);;
    const buf_344 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_345 = createEmptyBuf(device, 1310720);;
    const buf_346 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_347 = createEmptyBuf(device, 4194304);;
    const buf_348 = createEmptyBuf(device, 16384);;
    const buf_349 = createEmptyBuf(device, 16384);;
    const buf_350 = createEmptyBuf(device, 2097152);;
    const buf_351 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_352 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_353 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight']));
    const buf_354 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias']));
    const buf_355 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_356 = createEmptyBuf(device, 1261568);;
    const buf_357 = createEmptyBuf(device, 630784);;
    const buf_358 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_359 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_360 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight']));
    const buf_361 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias']));
    const buf_362 = createWeightBuf(device, 26214400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_363 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_364 = createWeightBuf(device, 13107200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_365 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_366 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.proj_out.weight']));
    const buf_367 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.7.1.proj_out.bias']));
    const buf_368 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.in_layers.0.weight']));
    const buf_369 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.in_layers.0.bias']));
    const buf_370 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.in_layers.2.weight']));
    const buf_371 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.in_layers.2.bias']));
    const buf_372 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.out_layers.0.weight']));
    const buf_373 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.out_layers.0.bias']));
    const buf_374 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.out_layers.3.weight']));
    const buf_375 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.0.out_layers.3.bias']));
    const buf_376 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.norm.weight']));
    const buf_377 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.norm.bias']));
    const buf_378 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.proj_in.weight']));
    const buf_379 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.proj_in.bias']));
    const buf_380 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight']));
    const buf_381 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias']));
    const buf_382 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_383 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_384 = createEmptyBuf(device, 1310720);;
    const buf_385 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_386 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_387 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_388 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight']));
    const buf_389 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias']));
    const buf_390 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_391 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_392 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_393 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight']));
    const buf_394 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias']));
    const buf_395 = createWeightBuf(device, 26214400, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_396 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_397 = createWeightBuf(device, 13107200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_398 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_399 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.proj_out.weight']));
    const buf_400 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.8.1.proj_out.bias']));
    const buf_401 = createEmptyBuf(device, 327680);;
    const buf_402 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.9.0.op.weight']));
    const buf_403 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.9.0.op.bias']));
    const buf_404 = createEmptyBuf(device, 327680);;
    const buf_405 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.in_layers.0.weight']));
    const buf_406 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.in_layers.0.bias']));
    const buf_407 = createEmptyBuf(device, 327680);;
    const buf_408 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.in_layers.2.weight']));
    const buf_409 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.in_layers.2.bias']));
    const buf_410 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.out_layers.0.weight']));
    const buf_411 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.out_layers.0.bias']));
    const buf_412 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.out_layers.3.weight']));
    const buf_413 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.10.0.out_layers.3.bias']));
    const buf_414 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.in_layers.0.weight']));
    const buf_415 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.in_layers.0.bias']));
    const buf_416 = createEmptyBuf(device, 327680);;
    const buf_417 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.in_layers.2.weight']));
    const buf_418 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.in_layers.2.bias']));
    const buf_419 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.out_layers.0.weight']));
    const buf_420 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.out_layers.0.bias']));
    const buf_421 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.out_layers.3.weight']));
    const buf_422 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.input_blocks.11.0.out_layers.3.bias']));
    const buf_423 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.in_layers.0.weight']));
    const buf_424 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.in_layers.0.bias']));
    const buf_425 = createEmptyBuf(device, 327680);;
    const buf_426 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.in_layers.2.weight']));
    const buf_427 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.in_layers.2.bias']));
    const buf_428 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.out_layers.0.weight']));
    const buf_429 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.out_layers.0.bias']));
    const buf_430 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.out_layers.3.weight']));
    const buf_431 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.0.out_layers.3.bias']));
    const buf_432 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.norm.weight']));
    const buf_433 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.norm.bias']));
    const buf_434 = createEmptyBuf(device, 327680);;
    const buf_435 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.proj_in.weight']));
    const buf_436 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.proj_in.bias']));
    const buf_437 = createEmptyBuf(device, 256);;
    const buf_438 = createEmptyBuf(device, 256);;
    const buf_439 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight']));
    const buf_440 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias']));
    const buf_441 = createEmptyBuf(device, 327680);;
    const buf_442 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_443 = createEmptyBuf(device, 327680);;
    const buf_444 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_445 = createEmptyBuf(device, 327680);;
    const buf_446 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_447 = createEmptyBuf(device, 4096);;
    const buf_448 = createEmptyBuf(device, 4096);;
    const buf_449 = createEmptyBuf(device, 131072);;
    const buf_450 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_451 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_452 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight']));
    const buf_453 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias']));
    const buf_454 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_455 = createEmptyBuf(device, 315392);;
    const buf_456 = createEmptyBuf(device, 157696);;
    const buf_457 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_458 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_459 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight']));
    const buf_460 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias']));
    const buf_461 = createWeightBuf(device, 26214400, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_462 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_463 = createWeightBuf(device, 13107200, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_464 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_465 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.proj_out.weight']));
    const buf_466 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.1.proj_out.bias']));
    const buf_467 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.in_layers.0.weight']));
    const buf_468 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.in_layers.0.bias']));
    const buf_469 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.in_layers.2.weight']));
    const buf_470 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.in_layers.2.bias']));
    const buf_471 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.out_layers.0.weight']));
    const buf_472 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.out_layers.0.bias']));
    const buf_473 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.out_layers.3.weight']));
    const buf_474 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.middle_block.2.out_layers.3.bias']));
    const buf_475 = createEmptyBuf(device, 655360);;
    const buf_476 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.in_layers.0.weight']));
    const buf_477 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.in_layers.0.bias']));
    const buf_478 = createWeightBuf(device, 58982400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.in_layers.2.weight']));
    const buf_479 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.in_layers.2.bias']));
    const buf_480 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.out_layers.0.weight']));
    const buf_481 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.out_layers.0.bias']));
    const buf_482 = createEmptyBuf(device, 655360);;
    const buf_483 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.out_layers.3.weight']));
    const buf_484 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.skip_connection.weight']));
    const buf_485 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.skip_connection.bias']));
    const buf_486 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.0.0.out_layers.3.bias']));
    const buf_487 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.in_layers.0.weight']));
    const buf_488 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.in_layers.0.bias']));
    const buf_489 = createWeightBuf(device, 58982400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.in_layers.2.weight']));
    const buf_490 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.in_layers.2.bias']));
    const buf_491 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.out_layers.0.weight']));
    const buf_492 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.out_layers.0.bias']));
    const buf_493 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.out_layers.3.weight']));
    const buf_494 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.skip_connection.weight']));
    const buf_495 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.skip_connection.bias']));
    const buf_496 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.1.0.out_layers.3.bias']));
    const buf_497 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.in_layers.0.weight']));
    const buf_498 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.in_layers.0.bias']));
    const buf_499 = createWeightBuf(device, 58982400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.in_layers.2.weight']));
    const buf_500 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.in_layers.2.bias']));
    const buf_501 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.out_layers.0.weight']));
    const buf_502 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.out_layers.0.bias']));
    const buf_503 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.out_layers.3.weight']));
    const buf_504 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.skip_connection.weight']));
    const buf_505 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.skip_connection.bias']));
    const buf_506 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.0.out_layers.3.bias']));
    const buf_507 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.1.conv.weight']));
    const buf_508 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.2.1.conv.bias']));
    const buf_509 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.in_layers.0.weight']));
    const buf_510 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.in_layers.0.bias']));
    const buf_511 = createWeightBuf(device, 58982400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.in_layers.2.weight']));
    const buf_512 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.in_layers.2.bias']));
    const buf_513 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.out_layers.0.weight']));
    const buf_514 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.out_layers.0.bias']));
    const buf_515 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.out_layers.3.weight']));
    const buf_516 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.skip_connection.weight']));
    const buf_517 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.skip_connection.bias']));
    const buf_518 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.0.out_layers.3.bias']));
    const buf_519 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.norm.weight']));
    const buf_520 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.norm.bias']));
    const buf_521 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.proj_in.weight']));
    const buf_522 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.proj_in.bias']));
    const buf_523 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight']));
    const buf_524 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias']));
    const buf_525 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_526 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_527 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_528 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_529 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_530 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight']));
    const buf_531 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias']));
    const buf_532 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_533 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_534 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_535 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight']));
    const buf_536 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias']));
    const buf_537 = createWeightBuf(device, 26214400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_538 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_539 = createWeightBuf(device, 13107200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_540 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_541 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.proj_out.weight']));
    const buf_542 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.3.1.proj_out.bias']));
    const buf_543 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.in_layers.0.weight']));
    const buf_544 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.in_layers.0.bias']));
    const buf_545 = createWeightBuf(device, 58982400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.in_layers.2.weight']));
    const buf_546 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.in_layers.2.bias']));
    const buf_547 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.out_layers.0.weight']));
    const buf_548 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.out_layers.0.bias']));
    const buf_549 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.out_layers.3.weight']));
    const buf_550 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.skip_connection.weight']));
    const buf_551 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.skip_connection.bias']));
    const buf_552 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.0.out_layers.3.bias']));
    const buf_553 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.norm.weight']));
    const buf_554 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.norm.bias']));
    const buf_555 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.proj_in.weight']));
    const buf_556 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.proj_in.bias']));
    const buf_557 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight']));
    const buf_558 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias']));
    const buf_559 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_560 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_561 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_562 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_563 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_564 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight']));
    const buf_565 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias']));
    const buf_566 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_567 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_568 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_569 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight']));
    const buf_570 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias']));
    const buf_571 = createWeightBuf(device, 26214400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_572 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_573 = createWeightBuf(device, 13107200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_574 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_575 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.proj_out.weight']));
    const buf_576 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.4.1.proj_out.bias']));
    const buf_577 = createEmptyBuf(device, 1966080);;
    const buf_578 = createEmptyBuf(device, 1966080);;
    const buf_579 = createWeightBuf(device, 3840, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.in_layers.0.weight']));
    const buf_580 = createWeightBuf(device, 3840, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.in_layers.0.bias']));
    const buf_581 = createWeightBuf(device, 44236800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.in_layers.2.weight']));
    const buf_582 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.in_layers.2.bias']));
    const buf_583 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.out_layers.0.weight']));
    const buf_584 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.out_layers.0.bias']));
    const buf_585 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.out_layers.3.weight']));
    const buf_586 = createWeightBuf(device, 4915200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.skip_connection.weight']));
    const buf_587 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.skip_connection.bias']));
    const buf_588 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.0.out_layers.3.bias']));
    const buf_589 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.norm.weight']));
    const buf_590 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.norm.bias']));
    const buf_591 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.proj_in.weight']));
    const buf_592 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.proj_in.bias']));
    const buf_593 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight']));
    const buf_594 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias']));
    const buf_595 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_596 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_597 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_598 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_599 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_600 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight']));
    const buf_601 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias']));
    const buf_602 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_603 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_604 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_605 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight']));
    const buf_606 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias']));
    const buf_607 = createWeightBuf(device, 26214400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_608 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_609 = createWeightBuf(device, 13107200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_610 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_611 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.proj_out.weight']));
    const buf_612 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.1.proj_out.bias']));
    const buf_613 = createWeightBuf(device, 29491200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.2.conv.weight']));
    const buf_614 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.5.2.conv.bias']));
    const buf_615 = createEmptyBuf(device, 7864320);;
    const buf_616 = createEmptyBuf(device, 7864320);;
    const buf_617 = createWeightBuf(device, 3840, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.in_layers.0.weight']));
    const buf_618 = createWeightBuf(device, 3840, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.in_layers.0.bias']));
    const buf_619 = createWeightBuf(device, 22118400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.in_layers.2.weight']));
    const buf_620 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.in_layers.2.bias']));
    const buf_621 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.out_layers.0.weight']));
    const buf_622 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.out_layers.0.bias']));
    const buf_623 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.out_layers.3.weight']));
    const buf_624 = createWeightBuf(device, 2457600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.skip_connection.weight']));
    const buf_625 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.skip_connection.bias']));
    const buf_626 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.0.out_layers.3.bias']));
    const buf_627 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.norm.weight']));
    const buf_628 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.norm.bias']));
    const buf_629 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.proj_in.weight']));
    const buf_630 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.proj_in.bias']));
    const buf_631 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.weight']));
    const buf_632 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.bias']));
    const buf_633 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_634 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_635 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_636 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_637 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_638 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.weight']));
    const buf_639 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.bias']));
    const buf_640 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_641 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_642 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_643 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.weight']));
    const buf_644 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.bias']));
    const buf_645 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_646 = createWeightBuf(device, 10240, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_647 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_648 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_649 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.proj_out.weight']));
    const buf_650 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.6.1.proj_out.bias']));
    const buf_651 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.in_layers.0.weight']));
    const buf_652 = createWeightBuf(device, 2560, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.in_layers.0.bias']));
    const buf_653 = createWeightBuf(device, 14745600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.in_layers.2.weight']));
    const buf_654 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.in_layers.2.bias']));
    const buf_655 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.out_layers.0.weight']));
    const buf_656 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.out_layers.0.bias']));
    const buf_657 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.out_layers.3.weight']));
    const buf_658 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.skip_connection.weight']));
    const buf_659 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.skip_connection.bias']));
    const buf_660 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.0.out_layers.3.bias']));
    const buf_661 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.norm.weight']));
    const buf_662 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.norm.bias']));
    const buf_663 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.proj_in.weight']));
    const buf_664 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.proj_in.bias']));
    const buf_665 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.weight']));
    const buf_666 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.bias']));
    const buf_667 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_668 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_669 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_670 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_671 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_672 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.weight']));
    const buf_673 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.bias']));
    const buf_674 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_675 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_676 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_677 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.weight']));
    const buf_678 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.bias']));
    const buf_679 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_680 = createWeightBuf(device, 10240, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_681 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_682 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_683 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.proj_out.weight']));
    const buf_684 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.7.1.proj_out.bias']));
    const buf_685 = createEmptyBuf(device, 3932160);;
    const buf_686 = createEmptyBuf(device, 3932160);;
    const buf_687 = createWeightBuf(device, 1920, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.in_layers.0.weight']));
    const buf_688 = createWeightBuf(device, 1920, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.in_layers.0.bias']));
    const buf_689 = createWeightBuf(device, 11059200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.in_layers.2.weight']));
    const buf_690 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.in_layers.2.bias']));
    const buf_691 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.out_layers.0.weight']));
    const buf_692 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.out_layers.0.bias']));
    const buf_693 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.out_layers.3.weight']));
    const buf_694 = createWeightBuf(device, 1228800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.skip_connection.weight']));
    const buf_695 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.skip_connection.bias']));
    const buf_696 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.0.out_layers.3.bias']));
    const buf_697 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.norm.weight']));
    const buf_698 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.norm.bias']));
    const buf_699 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.proj_in.weight']));
    const buf_700 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.proj_in.bias']));
    const buf_701 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.weight']));
    const buf_702 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.bias']));
    const buf_703 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_704 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_705 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_706 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_707 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_708 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.weight']));
    const buf_709 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.bias']));
    const buf_710 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_711 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_712 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_713 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight']));
    const buf_714 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.bias']));
    const buf_715 = createWeightBuf(device, 6553600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_716 = createWeightBuf(device, 10240, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_717 = createWeightBuf(device, 3276800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_718 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_719 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.proj_out.weight']));
    const buf_720 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.1.proj_out.bias']));
    const buf_721 = createWeightBuf(device, 7372800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.2.conv.weight']));
    const buf_722 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.8.2.conv.bias']));
    const buf_723 = createEmptyBuf(device, 15728640);;
    const buf_724 = createEmptyBuf(device, 15728640);;
    const buf_725 = createWeightBuf(device, 1920, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.in_layers.0.weight']));
    const buf_726 = createWeightBuf(device, 1920, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.in_layers.0.bias']));
    const buf_727 = createWeightBuf(device, 5529600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.in_layers.2.weight']));
    const buf_728 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.in_layers.2.bias']));
    const buf_729 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.out_layers.0.weight']));
    const buf_730 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.out_layers.0.bias']));
    const buf_731 = createEmptyBuf(device, 10485760);;
    const buf_732 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.out_layers.3.weight']));
    const buf_733 = createWeightBuf(device, 614400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.skip_connection.weight']));
    const buf_734 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.skip_connection.bias']));
    const buf_735 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.0.out_layers.3.bias']));
    const buf_736 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.norm.weight']));
    const buf_737 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.norm.bias']));
    const buf_738 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.proj_in.weight']));
    const buf_739 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.proj_in.bias']));
    const buf_740 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.weight']));
    const buf_741 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias']));
    const buf_742 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_743 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_744 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_745 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_746 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_747 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.weight']));
    const buf_748 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.bias']));
    const buf_749 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_750 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_751 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_752 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight']));
    const buf_753 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias']));
    const buf_754 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_755 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_756 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_757 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_758 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.proj_out.weight']));
    const buf_759 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.9.1.proj_out.bias']));
    const buf_760 = createEmptyBuf(device, 10485760);;
    const buf_761 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.in_layers.0.weight']));
    const buf_762 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.in_layers.0.bias']));
    const buf_763 = createWeightBuf(device, 3686400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.in_layers.2.weight']));
    const buf_764 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.in_layers.2.bias']));
    const buf_765 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.out_layers.0.weight']));
    const buf_766 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.out_layers.0.bias']));
    const buf_767 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.out_layers.3.weight']));
    const buf_768 = createWeightBuf(device, 409600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.skip_connection.weight']));
    const buf_769 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.skip_connection.bias']));
    const buf_770 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.0.out_layers.3.bias']));
    const buf_771 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.norm.weight']));
    const buf_772 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.norm.bias']));
    const buf_773 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.proj_in.weight']));
    const buf_774 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.proj_in.bias']));
    const buf_775 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.weight']));
    const buf_776 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.bias']));
    const buf_777 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_778 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_779 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_780 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_781 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_782 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.weight']));
    const buf_783 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.bias']));
    const buf_784 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_785 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_786 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_787 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.weight']));
    const buf_788 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.bias']));
    const buf_789 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_790 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_791 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_792 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_793 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.proj_out.weight']));
    const buf_794 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.10.1.proj_out.bias']));
    const buf_795 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.in_layers.0.weight']));
    const buf_796 = createWeightBuf(device, 1280, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.in_layers.0.bias']));
    const buf_797 = createWeightBuf(device, 3686400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.in_layers.2.weight']));
    const buf_798 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.in_layers.2.bias']));
    const buf_799 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.out_layers.0.weight']));
    const buf_800 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.out_layers.0.bias']));
    const buf_801 = createWeightBuf(device, 1843200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.out_layers.3.weight']));
    const buf_802 = createWeightBuf(device, 409600, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.skip_connection.weight']));
    const buf_803 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.skip_connection.bias']));
    const buf_804 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.0.out_layers.3.bias']));
    const buf_805 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.norm.weight']));
    const buf_806 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.norm.bias']));
    const buf_807 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.proj_in.weight']));
    const buf_808 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.proj_in.bias']));
    const buf_809 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.weight']));
    const buf_810 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias']));
    const buf_811 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.weight']));
    const buf_812 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight']));
    const buf_813 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.weight']));
    const buf_814 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.weight']));
    const buf_815 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.bias']));
    const buf_816 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.weight']));
    const buf_817 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.bias']));
    const buf_818 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight']));
    const buf_819 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight']));
    const buf_820 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias']));
    const buf_821 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.weight']));
    const buf_822 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.bias']));
    const buf_823 = createWeightBuf(device, 1638400, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.weight']));
    const buf_824 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.bias']));
    const buf_825 = createWeightBuf(device, 819200, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.weight']));
    const buf_826 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.bias']));
    const buf_827 = createWeightBuf(device, 204800, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.proj_out.weight']));
    const buf_828 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.output_blocks.11.1.proj_out.bias']));
    const buf_829 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.out.0.weight']));
    const buf_830 = createWeightBuf(device, 640, getTensorBuffer(safetensor, metadata['model.diffusion_model.out.0.bias']));
    const buf_831 = createEmptyBuf(device, 65536);;
    const buf_832 = createWeightBuf(device, 23040, getTensorBuffer(safetensor, metadata['model.diffusion_model.out.2.weight']));
    const buf_833 = createWeightBuf(device, 8, getTensorBuffer(safetensor, metadata['model.diffusion_model.out.2.bias']));
    const output0 = createEmptyBuf(device, 65536);;
    const input6 = createEmptyBuf(device, 4);;

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer1 = device.createBuffer({size:input1.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer2 = device.createBuffer({size:input2.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer3 = device.createBuffer({size:input3.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer4 = device.createBuffer({size:input4.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer5 = device.createBuffer({size:input5.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer6 = device.createBuffer({size:input6.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [E_n62, E_n63, E_128_32_4n2, r2_160_16_10, E_1848_32_2, E_n62, E_n63, r_2_80_8_8_16_4_4_4_3_3, E_5_16_4n1, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_20_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_10_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_77_5_2_16_192_4_4, r_512_32_40_4, r_1280_16_20, r_64_16_16, r_1280_16_80, r_8_4_8_16_40_4_4, r_320_16_80, r_320_16_80, r_640_16_80, r_640_16_80, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_1280_16_80n1, r_640_16_80, r_640_16_80, r_640_16_80, r_320_16_80, r_320_16_80, r_320_16_80, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3n1, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4n1, r_2_10_64_8_16_80_4_4_4, r_256_2_16_80_4, r_64_2_16_80_4_4, E_2_128_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_2_8_128_64_8_16_10_4_4_4, r_2048_32_1024_4, r_512_32_1024_4_4, E_8192_64_8_16_4, r_2_2_64_5_4_16_2_1024_4_4_4, r_2_128_5_8_16_80_4_4_4, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_2_64_77_8_16_10_4_4, r_2048_32_77, r_512_32_77_4, E_512_77_32_4, r_2_2_64_5_4_16_2_77_4_4, r_2_128_5_8_16_80_4_4_4n1, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_40_8_16_80_4_4_4, E_1024_20_8_16_4, r_256_5_8_16_320_4_4_4, r_2_10_64_8_16_80_4_4_4n1, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3n1, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4n1, r_2_10_64_8_16_80_4_4_4, r_256_2_16_80_4, r_64_2_16_80_4_4, E_2_128_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_2_8_128_64_8_16_10_4_4_4, r_2048_32_1024_4, r_512_32_1024_4_4, E_8192_64_8_16_4, r_2_2_64_5_4_16_2_1024_4_4_4, r_2_128_5_8_16_80_4_4_4, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_2_64_77_8_16_10_4_4, r_2048_32_77, r_512_32_77_4, E_512_77_32_4, r_2_2_64_5_4_16_2_77_4_4, r_2_128_5_8_16_80_4_4_4n1, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_40_8_16_80_4_4_4, E_1024_20_8_16_4, r_256_5_8_16_320_4_4_4, r_2_10_64_8_16_80_4_4_4n1, r_2_80_2_16_8_320_4_4_3_3, r_64_16_640, r_64_16_640n1, E_2_40_16_8_16_4, r_2_160_2_16_8_320_4_4_3_3, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4, r_2_160_2_16_8_640_4_4_3_3, r_2_20_16_8_16_80_4_4_4, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4n1, r_2_20_16_8_16_160_4_4_4, r_2_1024_16_40, r_2_1024_16_40n1, E_2_32_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_2_8_32_16_8_16_20_4_4_4, r_512_32_256_4, r_128_32_256_4_4, E_2048_16_8_16_4, r_2_4_16_5_2_16_4_256_4_4_4, r_2_32_10_8_16_160_4_4_4, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_2_16_77_8_16_20_4_4, r_512_32_77, r_128_32_77_4, E_128_77_32_4, r_2_4_16_5_2_16_4_77_4_4, r_2_32_10_8_16_160_4_4_4n1, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_80_8_16_160_4_4_4, E_256_40_8_16_4, r_64_10_8_16_640_4_4_4, r_2_20_16_8_16_160_4_4_4n1, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4, r_2_160_2_16_8_640_4_4_3_3n1, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4, r_2_160_2_16_8_640_4_4_3_3n2, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4n1, r_2_20_16_8_16_160_4_4_4, r_2_1024_16_40, r_2_1024_16_40n1, E_2_32_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_2_8_32_16_8_16_20_4_4_4, r_512_32_256_4, r_128_32_256_4_4, E_2048_16_8_16_4, r_2_4_16_5_2_16_4_256_4_4_4, r_2_32_10_8_16_160_4_4_4, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_2_16_77_8_16_20_4_4, r_512_32_77, r_128_32_77_4, E_128_77_32_4, r_2_4_16_5_2_16_4_77_4_4, r_2_32_10_8_16_160_4_4_4n1, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_80_8_16_160_4_4_4, E_256_40_8_16_4, r_64_10_8_16_640_4_4_4, r_2_20_16_8_16_160_4_4_4n1, r_2_80_2_16_4_640_4_4_3_3, r_64_16_320, r_64_16_320n1, E_2_80_4_8_16_4, r_2_160_2_16_4_640_4_4_3_3, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4, r_2_160_2_16_4_1280_4_4_3_3, r_2_40_4_8_16_160_4_4_4, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4n1, r_2_40_4_8_16_320_4_4_4, r_2_256_16_80, r_2_256_16_80n1, E_2_8_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_2_8_8_4_8_16_40_4_4_4, r_128_32_64_4, r_32_32_64_4_4, E_512_4_8_16_4, r_2_8_4_5_16_8_64_4_4_4, r_2_8_20_8_16_320_4_4_4, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_2_4_77_8_16_40_4_4, r_128_32_77, r_32_32_77_4, E_32_77_32_4, r_2_8_4_5_16_8_77_4_4, r_2_8_20_8_16_320_4_4_4n1, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_160_8_16_320_4_4_4, E_64_80_8_16_4, r_16_20_8_16_1280_4_4_4, r_2_40_4_8_16_320_4_4_4n1, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4, r_2_160_2_16_4_1280_4_4_3_3n1, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4, r_2_160_2_16_4_1280_4_4_3_3n2, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4n1, r_2_40_4_8_16_320_4_4_4, r_2_256_16_80, r_2_256_16_80n1, E_2_8_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_2_8_8_4_8_16_40_4_4_4, r_128_32_64_4, r_32_32_64_4_4, E_512_4_8_16_4, r_2_8_4_5_16_8_64_4_4_4, r_2_8_20_8_16_320_4_4_4, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_2_4_77_8_16_40_4_4, r_128_32_77, r_32_32_77_4, E_32_77_32_4, r_2_8_4_5_16_8_77_4_4, r_2_8_20_8_16_320_4_4_4n1, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_160_8_16_320_4_4_4, E_64_80_8_16_4, r_16_20_8_16_1280_4_4_4, r_2_40_4_8_16_320_4_4_4n1, r_2_40_8_8_2_1280_4_4_3_3, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n1, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n2, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n1, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n2, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n1, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n2, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4n1, r_2_40_8_16_320_4_4_4, r_2_64_16_80, r_2_64_16_80n1, E_2_2_20_8_16_4_4, r_4_20_8_16_320_4_4_4, r_4_20_8_16_320_4_4_4, r_4_20_8_16_320_4_4_4, r_2_8_2_8_16_40_4_4_4, r_1024_16_4, r_1024_16_4n1, E_128_8_16_4, r_2_8_5_16_8_16_4_4_4, r_2_2_20_8_16_320_4_4_4, r_128_16_80, r_128_16_80n1, E_4_20_8_16_4_4, r_4_20_8_16_320_4_4_4, r_2_77_8_16_40_4_4, r_32_32_77, r_8_32_77_4, E_8_77_32_4, r_2_8_5_16_8_77_4_4, r_2_2_20_8_16_320_4_4_4n1, r_128_16_80, r_128_16_80n1, E_4_20_8_16_4_4, r_4_160_8_16_320_4_4_4, E_16_80_8_16_4, r_4_20_8_16_1280_4_4_4, r_2_40_8_16_320_4_4_4n1, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n1, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n2, E_2_320_8_16_4, r_64_16_320, r_64_16_320n1, E_2_320_8_16_4n1, r_2_40_8_8_2_2560_4_4_3_3, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n3, r_2_40_8_16_640_4_4_4, E_2_320_8_16_4, r_64_16_320, r_64_16_320n1, E_2_320_8_16_4n1, r_2_40_8_8_2_2560_4_4_3_3, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n3, r_2_40_8_16_640_4_4_4, E_2_320_8_16_4, r_64_16_320, r_64_16_320n1, E_2_320_8_16_4n1, r_2_40_8_8_2_2560_4_4_3_3, r_64_16_160, r_64_16_160n1, E_2_160_8_16_4, r_2_40_8_8_2_1280_4_4_3_3n3, r_2_40_8_16_640_4_4_4, r_2_160_2_16_4_1280_4_4_3_3n3, E_2_320_4_8_16_4, r_64_16_1280, r_64_16_1280n1, E_2_320_4_8_16_4n1, r_2_160_2_16_4_2560_4_4_3_3, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4, r_2_160_2_16_4_1280_4_4_3_3, r_2_40_4_8_16_640_4_4_4, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4n1, r_2_40_4_8_16_320_4_4_4, r_2_256_16_80, r_2_256_16_80n1, E_2_8_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_2_8_8_4_8_16_40_4_4_4, r_128_32_64_4, r_32_32_64_4_4, E_512_4_8_16_4, r_2_8_4_5_16_8_64_4_4_4, r_2_8_20_8_16_320_4_4_4, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_2_4_77_8_16_40_4_4, r_128_32_77, r_32_32_77_4, E_32_77_32_4, r_2_8_4_5_16_8_77_4_4, r_2_8_20_8_16_320_4_4_4n1, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_160_8_16_320_4_4_4, E_64_80_8_16_4, r_16_20_8_16_1280_4_4_4, r_2_40_4_8_16_320_4_4_4n1, E_2_320_4_8_16_4, r_64_16_1280, r_64_16_1280n1, E_2_320_4_8_16_4n1, r_2_160_2_16_4_2560_4_4_3_3, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4, r_2_160_2_16_4_1280_4_4_3_3, r_2_40_4_8_16_640_4_4_4, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4n1, r_2_40_4_8_16_320_4_4_4, r_2_256_16_80, r_2_256_16_80n1, E_2_8_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_2_8_8_4_8_16_40_4_4_4, r_128_32_64_4, r_32_32_64_4_4, E_512_4_8_16_4, r_2_8_4_5_16_8_64_4_4_4, r_2_8_20_8_16_320_4_4_4, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_2_4_77_8_16_40_4_4, r_128_32_77, r_32_32_77_4, E_32_77_32_4, r_2_8_4_5_16_8_77_4_4, r_2_8_20_8_16_320_4_4_4n1, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_160_8_16_320_4_4_4, E_64_80_8_16_4, r_16_20_8_16_1280_4_4_4, r_2_40_4_8_16_320_4_4_4n1, E_2_240_4_8_16_4, r_64_16_960, r_64_16_960n1, E_2_240_4_8_16_4n1, r_2_160_2_16_4_1920_4_4_3_3, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4, r_2_160_2_16_4_1280_4_4_3_3, r_2_40_4_8_16_480_4_4_4, r_64_16_640, r_64_16_640n1, E_2_160_4_8_16_4n1, r_2_40_4_8_16_320_4_4_4, r_2_256_16_80, r_2_256_16_80n1, E_2_8_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_16_20_8_16_320_4_4_4, r_2_8_8_4_8_16_40_4_4_4, r_128_32_64_4, r_32_32_64_4_4, E_512_4_8_16_4, r_2_8_4_5_16_8_64_4_4_4, r_2_8_20_8_16_320_4_4_4, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_20_8_16_320_4_4_4, r_2_4_77_8_16_40_4_4, r_128_32_77, r_32_32_77_4, E_32_77_32_4, r_2_8_4_5_16_8_77_4_4, r_2_8_20_8_16_320_4_4_4n1, r_512_16_80, r_512_16_80n1, E_16_20_8_16_4_4, r_16_160_8_16_320_4_4_4, E_64_80_8_16_4, r_16_20_8_16_1280_4_4_4, r_2_40_4_8_16_320_4_4_4n1, r_2_320_2_16_8_1280_4_4_3_3, E_2_240_16_8_16_4, r_512_32_60_4, r_64_16_16n2, r_8_4_8_16_60_4_4, r_64_16_16n3, E_2_240_16_8_16_4n1, r_2_160_2_16_8_1920_4_4_3_3, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4, r_2_160_2_16_8_640_4_4_3_3, r_2_20_16_8_16_480_4_4_4, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4n1, r_2_20_16_8_16_160_4_4_4, r_2_1024_16_40, r_2_1024_16_40n1, E_2_32_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_2_8_32_16_8_16_20_4_4_4, r_512_32_256_4, r_128_32_256_4_4, E_2048_16_8_16_4, r_2_4_16_5_2_16_4_256_4_4_4, r_2_32_10_8_16_160_4_4_4, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_2_16_77_8_16_20_4_4, r_512_32_77, r_128_32_77_4, E_128_77_32_4, r_2_4_16_5_2_16_4_77_4_4, r_2_32_10_8_16_160_4_4_4n1, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_80_8_16_160_4_4_4, E_256_40_8_16_4, r_64_10_8_16_640_4_4_4, r_2_20_16_8_16_160_4_4_4n1, E_2_160_16_8_16_4, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_160_16_8_16_4n1, r_2_160_2_16_8_1280_4_4_3_3, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4, r_2_160_2_16_8_640_4_4_3_3, r_2_20_16_8_16_320_4_4_4, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4n1, r_2_20_16_8_16_160_4_4_4, r_2_1024_16_40, r_2_1024_16_40n1, E_2_32_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_2_8_32_16_8_16_20_4_4_4, r_512_32_256_4, r_128_32_256_4_4, E_2048_16_8_16_4, r_2_4_16_5_2_16_4_256_4_4_4, r_2_32_10_8_16_160_4_4_4, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_2_16_77_8_16_20_4_4, r_512_32_77, r_128_32_77_4, E_128_77_32_4, r_2_4_16_5_2_16_4_77_4_4, r_2_32_10_8_16_160_4_4_4n1, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_80_8_16_160_4_4_4, E_256_40_8_16_4, r_64_10_8_16_640_4_4_4, r_2_20_16_8_16_160_4_4_4n1, E_2_120_16_8_16_4, r_64_16_1920, r_64_16_1920n1, E_2_120_16_8_16_4n1, r_2_160_2_16_8_960_4_4_3_3, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4, r_2_160_2_16_8_640_4_4_3_3, r_2_20_16_8_16_240_4_4_4, r_64_16_1280, r_64_16_1280n1, E_2_80_16_8_16_4n1, r_2_20_16_8_16_160_4_4_4, r_2_1024_16_40, r_2_1024_16_40n1, E_2_32_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_64_10_8_16_160_4_4_4, r_2_8_32_16_8_16_20_4_4_4, r_512_32_256_4, r_128_32_256_4_4, E_2048_16_8_16_4, r_2_4_16_5_2_16_4_256_4_4_4, r_2_32_10_8_16_160_4_4_4, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_10_8_16_160_4_4_4, r_2_16_77_8_16_20_4_4, r_512_32_77, r_128_32_77_4, E_128_77_32_4, r_2_4_16_5_2_16_4_77_4_4, r_2_32_10_8_16_160_4_4_4n1, r_2048_16_40, r_2048_16_40n1, E_64_10_8_16_4_4, r_64_80_8_16_160_4_4_4, E_256_40_8_16_4, r_64_10_8_16_640_4_4_4, r_2_20_16_8_16_160_4_4_4n1, r_2_160_8_8_16_640_4_4_3_3, E_2_120_64_8_16_4, r_512_32_120_4, r_64_16_16n4, r_8_4_8_16_120_4_4, r_64_16_16n5, E_2_120_64_8_16_4n1, r_2_80_8_8_16_960_4_4_3_3, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3n2, r_2_10_64_8_16_240_4_4_4, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4n1, r_2_10_64_8_16_80_4_4_4, r_256_2_16_80_4, r_64_2_16_80_4_4, E_2_128_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_2_8_128_64_8_16_10_4_4_4, r_2048_32_1024_4, r_512_32_1024_4_4, E_8192_64_8_16_4, r_2_2_64_5_4_16_2_1024_4_4_4, r_2_128_5_8_16_80_4_4_4, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_2_64_77_8_16_10_4_4, r_2048_32_77, r_512_32_77_4, E_512_77_32_4, r_2_2_64_5_4_16_2_77_4_4, r_2_128_5_8_16_80_4_4_4n1, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_40_8_16_80_4_4_4, E_1024_20_8_16_4, r_256_5_8_16_320_4_4_4, r_2_10_64_8_16_80_4_4_4n1, E_2_80_64_8_16_4, r_512_32_80_4, r_64_16_16n6, r_8_4_8_16_80_4_4, r_64_16_16n7, E_2_80_64_8_16_4n1, r_2_80_8_8_16_640_4_4_3_3, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3n2, r_2_10_64_8_16_160_4_4_4, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4n1, r_2_10_64_8_16_80_4_4_4, r_256_2_16_80_4, r_64_2_16_80_4_4, E_2_128_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_2_8_128_64_8_16_10_4_4_4, r_2048_32_1024_4, r_512_32_1024_4_4, E_8192_64_8_16_4, r_2_2_64_5_4_16_2_1024_4_4_4, r_2_128_5_8_16_80_4_4_4, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_2_64_77_8_16_10_4_4, r_2048_32_77, r_512_32_77_4, E_512_77_32_4, r_2_2_64_5_4_16_2_77_4_4, r_2_128_5_8_16_80_4_4_4n1, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_40_8_16_80_4_4_4, E_1024_20_8_16_4, r_256_5_8_16_320_4_4_4, r_2_10_64_8_16_80_4_4_4n1, E_2_80_64_8_16_4, r_512_32_80_4, r_64_16_16n6, r_8_4_8_16_80_4_4, r_64_16_16n7, E_2_80_64_8_16_4n1, r_2_80_8_8_16_640_4_4_3_3, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_80_8_8_16_320_4_4_3_3n2, r_2_10_64_8_16_160_4_4_4, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4n1, r_2_10_64_8_16_80_4_4_4, r_256_2_16_80_4, r_64_2_16_80_4_4, E_2_128_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_256_5_8_16_80_4_4_4, r_2_8_128_64_8_16_10_4_4_4, r_2048_32_1024_4, r_512_32_1024_4_4, E_8192_64_8_16_4, r_2_2_64_5_4_16_2_1024_4_4_4, r_2_128_5_8_16_80_4_4_4, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_5_8_16_80_4_4_4, r_2_64_77_8_16_10_4_4, r_2048_32_77, r_512_32_77_4, E_512_77_32_4, r_2_2_64_5_4_16_2_77_4_4, r_2_128_5_8_16_80_4_4_4n1, r_256_32_80_4, r_64_32_80_4_4, E_256_5_8_16_4_4, r_256_40_8_16_80_4_4_4, E_1024_20_8_16_4, r_256_5_8_16_320_4_4_4, r_2_10_64_8_16_80_4_4_4n1, r_512_32_40_4, r_64_16_16, r_8_4_8_16_40_4_4, r_64_16_16n1, E_2_40_64_8_16_4, r_2_8_8_16_320_4_4_3_3, E_128_32_4n3];
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

    return async (_input0,_input1,_input2,_input3,_input4,_input5,_input6) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
    await gpuWriteBuffer1.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer1.getMappedRange()).set(_input1);
        gpuWriteBuffer1.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer1, 0, input1, 0, gpuWriteBuffer1.size);
    await gpuWriteBuffer2.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer2.getMappedRange()).set(_input2);
        gpuWriteBuffer2.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer2, 0, input2, 0, gpuWriteBuffer2.size);
    await gpuWriteBuffer3.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer3.getMappedRange()).set(_input3);
        gpuWriteBuffer3.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer3, 0, input3, 0, gpuWriteBuffer3.size);
    await gpuWriteBuffer4.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer4.getMappedRange()).set(_input4);
        gpuWriteBuffer4.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer4, 0, input4, 0, gpuWriteBuffer4.size);
    await gpuWriteBuffer5.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer5.getMappedRange()).set(_input5);
        gpuWriteBuffer5.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer5, 0, input5, 0, gpuWriteBuffer5.size);
    await gpuWriteBuffer6.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer6.getMappedRange()).set(_input6);
        gpuWriteBuffer6.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer6, 0, input6, 0, gpuWriteBuffer6.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input5], [1, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_1, input4], [1, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_2, input2], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_3, input3, buf_4], [160, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_5, input0, input1], [1848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_6, input4], [1, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_7, input5], [1, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_8, buf_2, buf_9, buf_10], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_11, buf_3, buf_4], [5, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_12, buf_5, buf_13], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_14, buf_5, buf_15], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_16, buf_5, buf_17], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_18, buf_5, buf_19], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_20, buf_5, buf_21], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_22, buf_5, buf_23], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_24, buf_5, buf_25], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_26, buf_5, buf_27], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_28, buf_5, buf_29], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_30, buf_5, buf_31], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_32, buf_5, buf_33], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_34, buf_5, buf_35], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_36, buf_5, buf_37], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_38, buf_5, buf_39], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_40, buf_5, buf_41], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_42, buf_5, buf_43], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_44, buf_5, buf_45], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_46, buf_5, buf_47], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_48, buf_5, buf_49], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_50, buf_5, buf_51], [20, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_52, buf_5, buf_53], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_54, buf_5, buf_55], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_56, buf_5, buf_57], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_58, buf_5, buf_59], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_60, buf_5, buf_61], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_62, buf_5, buf_63], [10, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_64, buf_5, buf_65], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_66, buf_5, buf_67], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_68, buf_5, buf_69], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_70, buf_5, buf_71], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_72, buf_5, buf_73], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_74, buf_5, buf_75], [5, 77, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_76, buf_8], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_77, buf_11, buf_78, buf_79], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_81, buf_77, buf_82, buf_83], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_76, buf_8, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_11, buf_81, buf_84, buf_85], [320, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_86, buf_81, buf_87, buf_88], [320, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_89, buf_81, buf_90, buf_91], [640, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_92, buf_81, buf_93, buf_94], [640, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_77, buf_81, buf_95, buf_96], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_97, buf_81, buf_98, buf_99], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_100, buf_81, buf_101, buf_102], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_103, buf_81, buf_104, buf_105], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_106, buf_81, buf_107, buf_108], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_109, buf_81, buf_110, buf_111], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_112, buf_81, buf_113, buf_114], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_115, buf_81, buf_116, buf_117], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_118, buf_81, buf_119, buf_120], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_121, buf_81, buf_122, buf_123], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_124, buf_81, buf_125, buf_126], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_127, buf_81, buf_128, buf_129], [1280, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_130, buf_81, buf_131, buf_132], [640, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_133, buf_81, buf_134, buf_135], [640, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_136, buf_81, buf_137, buf_138], [640, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_139, buf_81, buf_140, buf_141], [320, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_142, buf_81, buf_143, buf_144], [320, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_145, buf_81, buf_146, buf_147], [320, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_148, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_149, buf_8, buf_80, buf_148, buf_150, buf_151], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_152, buf_149, buf_153, buf_154, buf_11], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_76, buf_152], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_76, buf_152, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_148, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_149, buf_152, buf_80, buf_148, buf_155, buf_156], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_152, buf_8, buf_149, buf_157, buf_158], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_76, buf_152], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_76, buf_152, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_148, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_149, buf_152, buf_80, buf_148, buf_159, buf_160], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_161, buf_149, buf_162, buf_163], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_164, buf_161], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_165, buf_161, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_149, buf_161, buf_164, buf_165, buf_166, buf_167], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_168, buf_149, buf_169], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_170, buf_149, buf_171], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_172, buf_149, buf_173], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_174, buf_168, buf_170], [8192, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_175, buf_174], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_176, buf_174, buf_175], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_177, buf_174, buf_175, buf_176], [64, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_149, buf_177, buf_172], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_168, buf_161, buf_149, buf_178, buf_179], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_164, buf_168], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_165, buf_168, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_161, buf_168, buf_164, buf_165, buf_180, buf_181], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_170, buf_161, buf_182], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_183, buf_170, buf_12], [77, 64, 2]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_175, buf_183], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_176, buf_183, buf_175], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_184, buf_183, buf_175, buf_176], [77, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_172, buf_184, buf_14], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_149, buf_168, buf_172, buf_185, buf_186], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_164, buf_149], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_165, buf_149, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_168, buf_149, buf_164, buf_165, buf_187, buf_188], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_189, buf_168, buf_190, buf_191], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_192, buf_189], [20, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_161, buf_149, buf_192, buf_193, buf_194], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_170, buf_161, buf_195, buf_196, buf_152], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_76, buf_170], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_76, buf_170, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_148, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_152, buf_170, buf_80, buf_148, buf_197, buf_198], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_172, buf_152, buf_199, buf_200, buf_86], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_76, buf_172], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_76, buf_172, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_148, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_149, buf_172, buf_80, buf_148, buf_201, buf_202], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_168, buf_170, buf_149, buf_203, buf_204], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_76, buf_168], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_76, buf_168, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_148, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_161, buf_168, buf_80, buf_148, buf_205, buf_206], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_152, buf_161, buf_207, buf_208], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_164, buf_152], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_165, buf_152, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_172, buf_152, buf_164, buf_165, buf_209, buf_210], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_149, buf_172, buf_211], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_161, buf_172, buf_212], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_213, buf_172, buf_214], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_174, buf_149, buf_161], [8192, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_175, buf_174], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_176, buf_174, buf_175], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_177, buf_174, buf_175, buf_176], [64, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_172, buf_177, buf_213], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_149, buf_152, buf_172, buf_215, buf_216], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_164, buf_149], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_165, buf_149, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_152, buf_149, buf_164, buf_165, buf_217, buf_218], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_161, buf_152, buf_219], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_183, buf_161, buf_16], [77, 64, 2]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_175, buf_183], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_176, buf_183, buf_175], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_184, buf_183, buf_175, buf_176], [77, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_213, buf_184, buf_18], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_172, buf_149, buf_213, buf_220, buf_221], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_164, buf_172], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_165, buf_172, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_149, buf_172, buf_164, buf_165, buf_222, buf_223], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_189, buf_149, buf_224, buf_225], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_192, buf_189], [20, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_152, buf_172, buf_192, buf_226, buf_227], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_161, buf_152, buf_228, buf_229, buf_168], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_230, buf_161, buf_231, buf_232], [2, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_80, buf_230], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_148, buf_230, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_233, buf_230, buf_80, buf_148, buf_234, buf_235], [16, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_236, buf_233, buf_237, buf_238, buf_89], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_80, buf_236], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_148, buf_236, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_239, buf_236, buf_80, buf_148, buf_240, buf_241], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_242, buf_239, buf_243], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_236, buf_230, buf_244, buf_245, buf_242, buf_246], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_80, buf_236], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [buf_148, buf_236, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[171], layouts[171], infinityBuf, [buf_239, buf_236, buf_80, buf_148, buf_247, buf_248], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[172], layouts[172], infinityBuf, [buf_249, buf_239, buf_250, buf_251], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[173], layouts[173], infinityBuf, [buf_252, buf_249], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[174], layouts[174], infinityBuf, [buf_253, buf_249, buf_252], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[175], layouts[175], infinityBuf, [buf_239, buf_249, buf_252, buf_253, buf_254, buf_255], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[176], layouts[176], infinityBuf, [buf_256, buf_239, buf_257], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[177], layouts[177], infinityBuf, [buf_258, buf_239, buf_259], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[178], layouts[178], infinityBuf, [buf_260, buf_239, buf_261], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[179], layouts[179], infinityBuf, [buf_262, buf_256, buf_258], [512, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[180], layouts[180], infinityBuf, [buf_76, buf_262], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[181], layouts[181], infinityBuf, [buf_263, buf_262, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[182], layouts[182], infinityBuf, [buf_264, buf_262, buf_76, buf_263], [16, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[183], layouts[183], infinityBuf, [buf_239, buf_264, buf_260], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[184], layouts[184], infinityBuf, [buf_256, buf_249, buf_239, buf_265, buf_266], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[185], layouts[185], infinityBuf, [buf_252, buf_256], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[186], layouts[186], infinityBuf, [buf_253, buf_256, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[187], layouts[187], infinityBuf, [buf_249, buf_256, buf_252, buf_253, buf_267, buf_268], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[188], layouts[188], infinityBuf, [buf_258, buf_249, buf_269], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[189], layouts[189], infinityBuf, [buf_270, buf_258, buf_20], [77, 16, 2]);
        addComputePass(device, commandEncoder, pipelines[190], layouts[190], infinityBuf, [buf_76, buf_270], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[191], layouts[191], infinityBuf, [buf_263, buf_270, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[192], layouts[192], infinityBuf, [buf_271, buf_270, buf_76, buf_263], [77, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[193], layouts[193], infinityBuf, [buf_260, buf_271, buf_22], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[194], layouts[194], infinityBuf, [buf_239, buf_256, buf_260, buf_272, buf_273], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[195], layouts[195], infinityBuf, [buf_252, buf_239], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[196], layouts[196], infinityBuf, [buf_253, buf_239, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[197], layouts[197], infinityBuf, [buf_256, buf_239, buf_252, buf_253, buf_274, buf_275], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[198], layouts[198], infinityBuf, [buf_192, buf_256, buf_276, buf_277], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[199], layouts[199], infinityBuf, [buf_278, buf_192], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[200], layouts[200], infinityBuf, [buf_249, buf_239, buf_278, buf_279, buf_280], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[201], layouts[201], infinityBuf, [buf_258, buf_249, buf_281, buf_282, buf_236], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[202], layouts[202], infinityBuf, [buf_80, buf_258], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[203], layouts[203], infinityBuf, [buf_148, buf_258, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[204], layouts[204], infinityBuf, [buf_236, buf_258, buf_80, buf_148, buf_283, buf_284], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[205], layouts[205], infinityBuf, [buf_260, buf_236, buf_285, buf_286, buf_92], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[206], layouts[206], infinityBuf, [buf_80, buf_260], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[207], layouts[207], infinityBuf, [buf_148, buf_260, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[208], layouts[208], infinityBuf, [buf_239, buf_260, buf_80, buf_148, buf_287, buf_288], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[209], layouts[209], infinityBuf, [buf_256, buf_258, buf_239, buf_289, buf_290], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[210], layouts[210], infinityBuf, [buf_80, buf_256], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[211], layouts[211], infinityBuf, [buf_148, buf_256, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[212], layouts[212], infinityBuf, [buf_249, buf_256, buf_80, buf_148, buf_291, buf_292], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[213], layouts[213], infinityBuf, [buf_236, buf_249, buf_293, buf_294], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[214], layouts[214], infinityBuf, [buf_252, buf_236], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[215], layouts[215], infinityBuf, [buf_253, buf_236, buf_252], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[216], layouts[216], infinityBuf, [buf_260, buf_236, buf_252, buf_253, buf_295, buf_296], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[217], layouts[217], infinityBuf, [buf_239, buf_260, buf_297], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[218], layouts[218], infinityBuf, [buf_249, buf_260, buf_298], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[219], layouts[219], infinityBuf, [buf_299, buf_260, buf_300], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[220], layouts[220], infinityBuf, [buf_262, buf_239, buf_249], [512, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[221], layouts[221], infinityBuf, [buf_76, buf_262], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[222], layouts[222], infinityBuf, [buf_263, buf_262, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[223], layouts[223], infinityBuf, [buf_264, buf_262, buf_76, buf_263], [16, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[224], layouts[224], infinityBuf, [buf_260, buf_264, buf_299], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[225], layouts[225], infinityBuf, [buf_239, buf_236, buf_260, buf_301, buf_302], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[226], layouts[226], infinityBuf, [buf_252, buf_239], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[227], layouts[227], infinityBuf, [buf_253, buf_239, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[228], layouts[228], infinityBuf, [buf_236, buf_239, buf_252, buf_253, buf_303, buf_304], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[229], layouts[229], infinityBuf, [buf_249, buf_236, buf_305], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[230], layouts[230], infinityBuf, [buf_270, buf_249, buf_24], [77, 16, 2]);
        addComputePass(device, commandEncoder, pipelines[231], layouts[231], infinityBuf, [buf_76, buf_270], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[232], layouts[232], infinityBuf, [buf_263, buf_270, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[233], layouts[233], infinityBuf, [buf_271, buf_270, buf_76, buf_263], [77, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[234], layouts[234], infinityBuf, [buf_299, buf_271, buf_26], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[235], layouts[235], infinityBuf, [buf_260, buf_239, buf_299, buf_306, buf_307], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[236], layouts[236], infinityBuf, [buf_252, buf_260], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[237], layouts[237], infinityBuf, [buf_253, buf_260, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[238], layouts[238], infinityBuf, [buf_239, buf_260, buf_252, buf_253, buf_308, buf_309], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[239], layouts[239], infinityBuf, [buf_192, buf_239, buf_310, buf_311], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[240], layouts[240], infinityBuf, [buf_278, buf_192], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[241], layouts[241], infinityBuf, [buf_236, buf_260, buf_278, buf_312, buf_313], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[242], layouts[242], infinityBuf, [buf_249, buf_236, buf_314, buf_315, buf_256], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[243], layouts[243], infinityBuf, [buf_316, buf_249, buf_317, buf_318], [80, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[244], layouts[244], infinityBuf, [buf_80, buf_316], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[245], layouts[245], infinityBuf, [buf_148, buf_316, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[246], layouts[246], infinityBuf, [buf_319, buf_316, buf_80, buf_148, buf_320, buf_321], [4, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[247], layouts[247], infinityBuf, [buf_233, buf_319, buf_322, buf_323, buf_77], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[248], layouts[248], infinityBuf, [buf_80, buf_233], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[249], layouts[249], infinityBuf, [buf_148, buf_233, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[250], layouts[250], infinityBuf, [buf_324, buf_233, buf_80, buf_148, buf_325, buf_326], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[251], layouts[251], infinityBuf, [buf_327, buf_324, buf_328], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[252], layouts[252], infinityBuf, [buf_233, buf_316, buf_329, buf_330, buf_327, buf_331], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[253], layouts[253], infinityBuf, [buf_80, buf_233], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[254], layouts[254], infinityBuf, [buf_148, buf_233, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[255], layouts[255], infinityBuf, [buf_324, buf_233, buf_80, buf_148, buf_332, buf_333], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[256], layouts[256], infinityBuf, [buf_334, buf_324, buf_335, buf_336], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[257], layouts[257], infinityBuf, [buf_337, buf_334], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[258], layouts[258], infinityBuf, [buf_338, buf_334, buf_337], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[259], layouts[259], infinityBuf, [buf_324, buf_334, buf_337, buf_338, buf_339, buf_340], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[260], layouts[260], infinityBuf, [buf_341, buf_324, buf_342], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[261], layouts[261], infinityBuf, [buf_343, buf_324, buf_344], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[262], layouts[262], infinityBuf, [buf_345, buf_324, buf_346], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[263], layouts[263], infinityBuf, [buf_347, buf_341, buf_343], [32, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[264], layouts[264], infinityBuf, [buf_348, buf_347], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[265], layouts[265], infinityBuf, [buf_349, buf_347, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[266], layouts[266], infinityBuf, [buf_350, buf_347, buf_348, buf_349], [4, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[267], layouts[267], infinityBuf, [buf_324, buf_350, buf_345], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[268], layouts[268], infinityBuf, [buf_341, buf_334, buf_324, buf_351, buf_352], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[269], layouts[269], infinityBuf, [buf_337, buf_341], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[270], layouts[270], infinityBuf, [buf_338, buf_341, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[271], layouts[271], infinityBuf, [buf_334, buf_341, buf_337, buf_338, buf_353, buf_354], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[272], layouts[272], infinityBuf, [buf_343, buf_334, buf_355], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[273], layouts[273], infinityBuf, [buf_356, buf_343, buf_28], [77, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[274], layouts[274], infinityBuf, [buf_348, buf_356], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[275], layouts[275], infinityBuf, [buf_349, buf_356, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[276], layouts[276], infinityBuf, [buf_357, buf_356, buf_348, buf_349], [77, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[277], layouts[277], infinityBuf, [buf_345, buf_357, buf_30], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[278], layouts[278], infinityBuf, [buf_324, buf_341, buf_345, buf_358, buf_359], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[279], layouts[279], infinityBuf, [buf_337, buf_324], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[280], layouts[280], infinityBuf, [buf_338, buf_324, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[281], layouts[281], infinityBuf, [buf_341, buf_324, buf_337, buf_338, buf_360, buf_361], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[282], layouts[282], infinityBuf, [buf_278, buf_341, buf_362, buf_363], [160, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[283], layouts[283], infinityBuf, [buf_168, buf_278], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[284], layouts[284], infinityBuf, [buf_334, buf_324, buf_168, buf_364, buf_365], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[285], layouts[285], infinityBuf, [buf_343, buf_334, buf_366, buf_367, buf_233], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[286], layouts[286], infinityBuf, [buf_80, buf_343], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[287], layouts[287], infinityBuf, [buf_148, buf_343, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[288], layouts[288], infinityBuf, [buf_233, buf_343, buf_80, buf_148, buf_368, buf_369], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[289], layouts[289], infinityBuf, [buf_345, buf_233, buf_370, buf_371, buf_97], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[290], layouts[290], infinityBuf, [buf_80, buf_345], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[291], layouts[291], infinityBuf, [buf_148, buf_345, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[292], layouts[292], infinityBuf, [buf_324, buf_345, buf_80, buf_148, buf_372, buf_373], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[293], layouts[293], infinityBuf, [buf_341, buf_343, buf_324, buf_374, buf_375], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[294], layouts[294], infinityBuf, [buf_80, buf_341], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[295], layouts[295], infinityBuf, [buf_148, buf_341, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[296], layouts[296], infinityBuf, [buf_334, buf_341, buf_80, buf_148, buf_376, buf_377], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[297], layouts[297], infinityBuf, [buf_233, buf_334, buf_378, buf_379], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[298], layouts[298], infinityBuf, [buf_337, buf_233], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[299], layouts[299], infinityBuf, [buf_338, buf_233, buf_337], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[300], layouts[300], infinityBuf, [buf_345, buf_233, buf_337, buf_338, buf_380, buf_381], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[301], layouts[301], infinityBuf, [buf_324, buf_345, buf_382], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[302], layouts[302], infinityBuf, [buf_334, buf_345, buf_383], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[303], layouts[303], infinityBuf, [buf_384, buf_345, buf_385], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[304], layouts[304], infinityBuf, [buf_347, buf_324, buf_334], [32, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[305], layouts[305], infinityBuf, [buf_348, buf_347], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[306], layouts[306], infinityBuf, [buf_349, buf_347, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[307], layouts[307], infinityBuf, [buf_350, buf_347, buf_348, buf_349], [4, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[308], layouts[308], infinityBuf, [buf_345, buf_350, buf_384], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[309], layouts[309], infinityBuf, [buf_324, buf_233, buf_345, buf_386, buf_387], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[310], layouts[310], infinityBuf, [buf_337, buf_324], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[311], layouts[311], infinityBuf, [buf_338, buf_324, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[312], layouts[312], infinityBuf, [buf_233, buf_324, buf_337, buf_338, buf_388, buf_389], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[313], layouts[313], infinityBuf, [buf_334, buf_233, buf_390], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[314], layouts[314], infinityBuf, [buf_356, buf_334, buf_32], [77, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[315], layouts[315], infinityBuf, [buf_348, buf_356], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[316], layouts[316], infinityBuf, [buf_349, buf_356, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[317], layouts[317], infinityBuf, [buf_357, buf_356, buf_348, buf_349], [77, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[318], layouts[318], infinityBuf, [buf_384, buf_357, buf_34], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[319], layouts[319], infinityBuf, [buf_345, buf_324, buf_384, buf_391, buf_392], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[320], layouts[320], infinityBuf, [buf_337, buf_345], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[321], layouts[321], infinityBuf, [buf_338, buf_345, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[322], layouts[322], infinityBuf, [buf_324, buf_345, buf_337, buf_338, buf_393, buf_394], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[323], layouts[323], infinityBuf, [buf_278, buf_324, buf_395, buf_396], [160, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[324], layouts[324], infinityBuf, [buf_213, buf_278], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[325], layouts[325], infinityBuf, [buf_233, buf_345, buf_213, buf_397, buf_398], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[326], layouts[326], infinityBuf, [buf_334, buf_233, buf_399, buf_400, buf_341], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[327], layouts[327], infinityBuf, [buf_401, buf_334, buf_402, buf_403], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[328], layouts[328], infinityBuf, [buf_80, buf_401], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[329], layouts[329], infinityBuf, [buf_148, buf_401, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[330], layouts[330], infinityBuf, [buf_404, buf_401, buf_80, buf_148, buf_405, buf_406], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[331], layouts[331], infinityBuf, [buf_407, buf_404, buf_408, buf_409, buf_100], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[332], layouts[332], infinityBuf, [buf_80, buf_407], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[333], layouts[333], infinityBuf, [buf_148, buf_407, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[334], layouts[334], infinityBuf, [buf_404, buf_407, buf_80, buf_148, buf_410, buf_411], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[335], layouts[335], infinityBuf, [buf_407, buf_401, buf_404, buf_412, buf_413], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[336], layouts[336], infinityBuf, [buf_80, buf_407], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[337], layouts[337], infinityBuf, [buf_148, buf_407, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[338], layouts[338], infinityBuf, [buf_404, buf_407, buf_80, buf_148, buf_414, buf_415], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[339], layouts[339], infinityBuf, [buf_416, buf_404, buf_417, buf_418, buf_103], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[340], layouts[340], infinityBuf, [buf_80, buf_416], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[341], layouts[341], infinityBuf, [buf_148, buf_416, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[342], layouts[342], infinityBuf, [buf_404, buf_416, buf_80, buf_148, buf_419, buf_420], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[343], layouts[343], infinityBuf, [buf_416, buf_407, buf_404, buf_421, buf_422], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[344], layouts[344], infinityBuf, [buf_80, buf_416], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[345], layouts[345], infinityBuf, [buf_148, buf_416, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[346], layouts[346], infinityBuf, [buf_404, buf_416, buf_80, buf_148, buf_423, buf_424], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[347], layouts[347], infinityBuf, [buf_425, buf_404, buf_426, buf_427, buf_106], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[348], layouts[348], infinityBuf, [buf_80, buf_425], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[349], layouts[349], infinityBuf, [buf_148, buf_425, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[350], layouts[350], infinityBuf, [buf_404, buf_425, buf_80, buf_148, buf_428, buf_429], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[351], layouts[351], infinityBuf, [buf_425, buf_416, buf_404, buf_430, buf_431], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[352], layouts[352], infinityBuf, [buf_80, buf_425], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[353], layouts[353], infinityBuf, [buf_148, buf_425, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[354], layouts[354], infinityBuf, [buf_404, buf_425, buf_80, buf_148, buf_432, buf_433], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[355], layouts[355], infinityBuf, [buf_434, buf_404, buf_435, buf_436], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[356], layouts[356], infinityBuf, [buf_437, buf_434], [64, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[357], layouts[357], infinityBuf, [buf_438, buf_434, buf_437], [64, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[358], layouts[358], infinityBuf, [buf_404, buf_434, buf_437, buf_438, buf_439, buf_440], [20, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[359], layouts[359], infinityBuf, [buf_441, buf_404, buf_442], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[360], layouts[360], infinityBuf, [buf_443, buf_404, buf_444], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[361], layouts[361], infinityBuf, [buf_445, buf_404, buf_446], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[362], layouts[362], infinityBuf, [buf_175, buf_441, buf_443], [2, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[363], layouts[363], infinityBuf, [buf_447, buf_175], [1024, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[364], layouts[364], infinityBuf, [buf_448, buf_175, buf_447], [1024, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[365], layouts[365], infinityBuf, [buf_449, buf_175, buf_447, buf_448], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[366], layouts[366], infinityBuf, [buf_404, buf_449, buf_445], [5, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[367], layouts[367], infinityBuf, [buf_441, buf_434, buf_404, buf_450, buf_451], [20, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[368], layouts[368], infinityBuf, [buf_437, buf_441], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[369], layouts[369], infinityBuf, [buf_438, buf_441, buf_437], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[370], layouts[370], infinityBuf, [buf_434, buf_441, buf_437, buf_438, buf_452, buf_453], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[371], layouts[371], infinityBuf, [buf_443, buf_434, buf_454], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[372], layouts[372], infinityBuf, [buf_455, buf_443, buf_36], [77, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[373], layouts[373], infinityBuf, [buf_447, buf_455], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[374], layouts[374], infinityBuf, [buf_448, buf_455, buf_447], [8, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[375], layouts[375], infinityBuf, [buf_456, buf_455, buf_447, buf_448], [77, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[376], layouts[376], infinityBuf, [buf_445, buf_456, buf_38], [5, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[377], layouts[377], infinityBuf, [buf_404, buf_441, buf_445, buf_457, buf_458], [20, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[378], layouts[378], infinityBuf, [buf_437, buf_404], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[379], layouts[379], infinityBuf, [buf_438, buf_404, buf_437], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[380], layouts[380], infinityBuf, [buf_441, buf_404, buf_437, buf_438, buf_459, buf_460], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[381], layouts[381], infinityBuf, [buf_256, buf_441, buf_461, buf_462], [160, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[382], layouts[382], infinityBuf, [buf_341, buf_256], [80, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[383], layouts[383], infinityBuf, [buf_434, buf_404, buf_341, buf_463, buf_464], [20, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[384], layouts[384], infinityBuf, [buf_443, buf_434, buf_465, buf_466, buf_425], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[385], layouts[385], infinityBuf, [buf_80, buf_443], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[386], layouts[386], infinityBuf, [buf_148, buf_443, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[387], layouts[387], infinityBuf, [buf_425, buf_443, buf_80, buf_148, buf_467, buf_468], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[388], layouts[388], infinityBuf, [buf_445, buf_425, buf_469, buf_470, buf_109], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[389], layouts[389], infinityBuf, [buf_80, buf_445], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[390], layouts[390], infinityBuf, [buf_148, buf_445, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[391], layouts[391], infinityBuf, [buf_404, buf_445, buf_80, buf_148, buf_471, buf_472], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[392], layouts[392], infinityBuf, [buf_441, buf_443, buf_404, buf_473, buf_474], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[393], layouts[393], infinityBuf, [buf_319, buf_441, buf_416], [320, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[394], layouts[394], infinityBuf, [buf_80, buf_319], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[395], layouts[395], infinityBuf, [buf_148, buf_319, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[396], layouts[396], infinityBuf, [buf_475, buf_319, buf_80, buf_148, buf_476, buf_477], [320, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[397], layouts[397], infinityBuf, [buf_416, buf_475, buf_478, buf_479, buf_112], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[398], layouts[398], infinityBuf, [buf_80, buf_416], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[399], layouts[399], infinityBuf, [buf_148, buf_416, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[400], layouts[400], infinityBuf, [buf_434, buf_416, buf_80, buf_148, buf_480, buf_481], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[401], layouts[401], infinityBuf, [buf_482, buf_434, buf_483], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[402], layouts[402], infinityBuf, [buf_443, buf_319, buf_484, buf_485, buf_482, buf_486], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[403], layouts[403], infinityBuf, [buf_319, buf_443, buf_407], [320, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[404], layouts[404], infinityBuf, [buf_80, buf_319], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[405], layouts[405], infinityBuf, [buf_148, buf_319, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[406], layouts[406], infinityBuf, [buf_475, buf_319, buf_80, buf_148, buf_487, buf_488], [320, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[407], layouts[407], infinityBuf, [buf_407, buf_475, buf_489, buf_490, buf_115], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[408], layouts[408], infinityBuf, [buf_80, buf_407], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[409], layouts[409], infinityBuf, [buf_148, buf_407, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[410], layouts[410], infinityBuf, [buf_425, buf_407, buf_80, buf_148, buf_491, buf_492], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[411], layouts[411], infinityBuf, [buf_482, buf_425, buf_493], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[412], layouts[412], infinityBuf, [buf_445, buf_319, buf_494, buf_495, buf_482, buf_496], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[413], layouts[413], infinityBuf, [buf_319, buf_445, buf_401], [320, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[414], layouts[414], infinityBuf, [buf_80, buf_319], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[415], layouts[415], infinityBuf, [buf_148, buf_319, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[416], layouts[416], infinityBuf, [buf_475, buf_319, buf_80, buf_148, buf_497, buf_498], [320, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[417], layouts[417], infinityBuf, [buf_401, buf_475, buf_499, buf_500, buf_118], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[418], layouts[418], infinityBuf, [buf_80, buf_401], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[419], layouts[419], infinityBuf, [buf_148, buf_401, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[420], layouts[420], infinityBuf, [buf_404, buf_401, buf_80, buf_148, buf_501, buf_502], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[421], layouts[421], infinityBuf, [buf_482, buf_404, buf_503], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[422], layouts[422], infinityBuf, [buf_441, buf_319, buf_504, buf_505, buf_482, buf_506], [40, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[423], layouts[423], infinityBuf, [buf_384, buf_441, buf_507, buf_508], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[424], layouts[424], infinityBuf, [buf_299, buf_384, buf_334], [4, 320, 2]);
        addComputePass(device, commandEncoder, pipelines[425], layouts[425], infinityBuf, [buf_80, buf_299], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[426], layouts[426], infinityBuf, [buf_148, buf_299, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[427], layouts[427], infinityBuf, [buf_260, buf_299, buf_80, buf_148, buf_509, buf_510], [4, 320, 2]);
        addComputePass(device, commandEncoder, pipelines[428], layouts[428], infinityBuf, [buf_345, buf_260, buf_511, buf_512, buf_121], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[429], layouts[429], infinityBuf, [buf_80, buf_345], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[430], layouts[430], infinityBuf, [buf_148, buf_345, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[431], layouts[431], infinityBuf, [buf_324, buf_345, buf_80, buf_148, buf_513, buf_514], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[432], layouts[432], infinityBuf, [buf_327, buf_324, buf_515], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[433], layouts[433], infinityBuf, [buf_233, buf_299, buf_516, buf_517, buf_327, buf_518], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[434], layouts[434], infinityBuf, [buf_80, buf_233], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[435], layouts[435], infinityBuf, [buf_148, buf_233, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[436], layouts[436], infinityBuf, [buf_334, buf_233, buf_80, buf_148, buf_519, buf_520], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[437], layouts[437], infinityBuf, [buf_341, buf_334, buf_521, buf_522], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[438], layouts[438], infinityBuf, [buf_337, buf_341], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[439], layouts[439], infinityBuf, [buf_338, buf_341, buf_337], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[440], layouts[440], infinityBuf, [buf_384, buf_341, buf_337, buf_338, buf_523, buf_524], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[441], layouts[441], infinityBuf, [buf_345, buf_384, buf_525], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[442], layouts[442], infinityBuf, [buf_324, buf_384, buf_526], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[443], layouts[443], infinityBuf, [buf_334, buf_384, buf_527], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[444], layouts[444], infinityBuf, [buf_347, buf_345, buf_324], [32, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[445], layouts[445], infinityBuf, [buf_348, buf_347], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[446], layouts[446], infinityBuf, [buf_349, buf_347, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[447], layouts[447], infinityBuf, [buf_350, buf_347, buf_348, buf_349], [4, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[448], layouts[448], infinityBuf, [buf_384, buf_350, buf_334], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[449], layouts[449], infinityBuf, [buf_345, buf_341, buf_384, buf_528, buf_529], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[450], layouts[450], infinityBuf, [buf_337, buf_345], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[451], layouts[451], infinityBuf, [buf_338, buf_345, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[452], layouts[452], infinityBuf, [buf_341, buf_345, buf_337, buf_338, buf_530, buf_531], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[453], layouts[453], infinityBuf, [buf_324, buf_341, buf_532], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[454], layouts[454], infinityBuf, [buf_356, buf_324, buf_40], [77, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[455], layouts[455], infinityBuf, [buf_348, buf_356], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[456], layouts[456], infinityBuf, [buf_349, buf_356, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[457], layouts[457], infinityBuf, [buf_357, buf_356, buf_348, buf_349], [77, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[458], layouts[458], infinityBuf, [buf_334, buf_357, buf_42], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[459], layouts[459], infinityBuf, [buf_384, buf_345, buf_334, buf_533, buf_534], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[460], layouts[460], infinityBuf, [buf_337, buf_384], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[461], layouts[461], infinityBuf, [buf_338, buf_384, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[462], layouts[462], infinityBuf, [buf_345, buf_384, buf_337, buf_338, buf_535, buf_536], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[463], layouts[463], infinityBuf, [buf_278, buf_345, buf_537, buf_538], [160, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[464], layouts[464], infinityBuf, [buf_172, buf_278], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[465], layouts[465], infinityBuf, [buf_341, buf_384, buf_172, buf_539, buf_540], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[466], layouts[466], infinityBuf, [buf_324, buf_341, buf_541, buf_542, buf_233], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[467], layouts[467], infinityBuf, [buf_239, buf_324, buf_343], [4, 320, 2]);
        addComputePass(device, commandEncoder, pipelines[468], layouts[468], infinityBuf, [buf_80, buf_239], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[469], layouts[469], infinityBuf, [buf_148, buf_239, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[470], layouts[470], infinityBuf, [buf_236, buf_239, buf_80, buf_148, buf_543, buf_544], [4, 320, 2]);
        addComputePass(device, commandEncoder, pipelines[471], layouts[471], infinityBuf, [buf_343, buf_236, buf_545, buf_546, buf_124], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[472], layouts[472], infinityBuf, [buf_80, buf_343], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[473], layouts[473], infinityBuf, [buf_148, buf_343, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[474], layouts[474], infinityBuf, [buf_233, buf_343, buf_80, buf_148, buf_547, buf_548], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[475], layouts[475], infinityBuf, [buf_327, buf_233, buf_549], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[476], layouts[476], infinityBuf, [buf_334, buf_239, buf_550, buf_551, buf_327, buf_552], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[477], layouts[477], infinityBuf, [buf_80, buf_334], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[478], layouts[478], infinityBuf, [buf_148, buf_334, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[479], layouts[479], infinityBuf, [buf_384, buf_334, buf_80, buf_148, buf_553, buf_554], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[480], layouts[480], infinityBuf, [buf_345, buf_384, buf_555, buf_556], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[481], layouts[481], infinityBuf, [buf_337, buf_345], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[482], layouts[482], infinityBuf, [buf_338, buf_345, buf_337], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[483], layouts[483], infinityBuf, [buf_341, buf_345, buf_337, buf_338, buf_557, buf_558], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[484], layouts[484], infinityBuf, [buf_324, buf_341, buf_559], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[485], layouts[485], infinityBuf, [buf_343, buf_341, buf_560], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[486], layouts[486], infinityBuf, [buf_233, buf_341, buf_561], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[487], layouts[487], infinityBuf, [buf_347, buf_324, buf_343], [32, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[488], layouts[488], infinityBuf, [buf_348, buf_347], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[489], layouts[489], infinityBuf, [buf_349, buf_347, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[490], layouts[490], infinityBuf, [buf_350, buf_347, buf_348, buf_349], [4, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[491], layouts[491], infinityBuf, [buf_384, buf_350, buf_233], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[492], layouts[492], infinityBuf, [buf_341, buf_345, buf_384, buf_562, buf_563], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[493], layouts[493], infinityBuf, [buf_337, buf_341], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[494], layouts[494], infinityBuf, [buf_338, buf_341, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[495], layouts[495], infinityBuf, [buf_345, buf_341, buf_337, buf_338, buf_564, buf_565], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[496], layouts[496], infinityBuf, [buf_324, buf_345, buf_566], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[497], layouts[497], infinityBuf, [buf_356, buf_324, buf_44], [77, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[498], layouts[498], infinityBuf, [buf_348, buf_356], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[499], layouts[499], infinityBuf, [buf_349, buf_356, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[500], layouts[500], infinityBuf, [buf_357, buf_356, buf_348, buf_349], [77, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[501], layouts[501], infinityBuf, [buf_343, buf_357, buf_46], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[502], layouts[502], infinityBuf, [buf_233, buf_341, buf_343, buf_567, buf_568], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[503], layouts[503], infinityBuf, [buf_337, buf_233], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[504], layouts[504], infinityBuf, [buf_338, buf_233, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[505], layouts[505], infinityBuf, [buf_384, buf_233, buf_337, buf_338, buf_569, buf_570], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[506], layouts[506], infinityBuf, [buf_278, buf_384, buf_571, buf_572], [160, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[507], layouts[507], infinityBuf, [buf_149, buf_278], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[508], layouts[508], infinityBuf, [buf_341, buf_233, buf_149, buf_573, buf_574], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[509], layouts[509], infinityBuf, [buf_345, buf_341, buf_575, buf_576, buf_334], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[510], layouts[510], infinityBuf, [buf_577, buf_345, buf_316], [4, 240, 2]);
        addComputePass(device, commandEncoder, pipelines[511], layouts[511], infinityBuf, [buf_80, buf_577], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[512], layouts[512], infinityBuf, [buf_148, buf_577, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[513], layouts[513], infinityBuf, [buf_578, buf_577, buf_80, buf_148, buf_579, buf_580], [4, 240, 2]);
        addComputePass(device, commandEncoder, pipelines[514], layouts[514], infinityBuf, [buf_334, buf_578, buf_581, buf_582, buf_127], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[515], layouts[515], infinityBuf, [buf_80, buf_334], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[516], layouts[516], infinityBuf, [buf_148, buf_334, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[517], layouts[517], infinityBuf, [buf_324, buf_334, buf_80, buf_148, buf_583, buf_584], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[518], layouts[518], infinityBuf, [buf_327, buf_324, buf_585], [160, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[519], layouts[519], infinityBuf, [buf_343, buf_577, buf_586, buf_587, buf_327, buf_588], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[520], layouts[520], infinityBuf, [buf_80, buf_343], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[521], layouts[521], infinityBuf, [buf_148, buf_343, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[522], layouts[522], infinityBuf, [buf_233, buf_343, buf_80, buf_148, buf_589, buf_590], [4, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[523], layouts[523], infinityBuf, [buf_384, buf_233, buf_591, buf_592], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[524], layouts[524], infinityBuf, [buf_337, buf_384], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[525], layouts[525], infinityBuf, [buf_338, buf_384, buf_337], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[526], layouts[526], infinityBuf, [buf_341, buf_384, buf_337, buf_338, buf_593, buf_594], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[527], layouts[527], infinityBuf, [buf_345, buf_341, buf_595], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[528], layouts[528], infinityBuf, [buf_334, buf_341, buf_596], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[529], layouts[529], infinityBuf, [buf_324, buf_341, buf_597], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[530], layouts[530], infinityBuf, [buf_347, buf_345, buf_334], [32, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[531], layouts[531], infinityBuf, [buf_348, buf_347], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[532], layouts[532], infinityBuf, [buf_349, buf_347, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[533], layouts[533], infinityBuf, [buf_350, buf_347, buf_348, buf_349], [4, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[534], layouts[534], infinityBuf, [buf_233, buf_350, buf_324], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[535], layouts[535], infinityBuf, [buf_341, buf_384, buf_233, buf_598, buf_599], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[536], layouts[536], infinityBuf, [buf_337, buf_341], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[537], layouts[537], infinityBuf, [buf_338, buf_341, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[538], layouts[538], infinityBuf, [buf_384, buf_341, buf_337, buf_338, buf_600, buf_601], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[539], layouts[539], infinityBuf, [buf_345, buf_384, buf_602], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[540], layouts[540], infinityBuf, [buf_356, buf_345, buf_48], [77, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[541], layouts[541], infinityBuf, [buf_348, buf_356], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[542], layouts[542], infinityBuf, [buf_349, buf_356, buf_348], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[543], layouts[543], infinityBuf, [buf_357, buf_356, buf_348, buf_349], [77, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[544], layouts[544], infinityBuf, [buf_334, buf_357, buf_50], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[545], layouts[545], infinityBuf, [buf_324, buf_341, buf_334, buf_603, buf_604], [20, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[546], layouts[546], infinityBuf, [buf_337, buf_324], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[547], layouts[547], infinityBuf, [buf_338, buf_324, buf_337], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[548], layouts[548], infinityBuf, [buf_233, buf_324, buf_337, buf_338, buf_605, buf_606], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[549], layouts[549], infinityBuf, [buf_278, buf_233, buf_607, buf_608], [160, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[550], layouts[550], infinityBuf, [buf_152, buf_278], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[551], layouts[551], infinityBuf, [buf_341, buf_324, buf_152, buf_609, buf_610], [20, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[552], layouts[552], infinityBuf, [buf_384, buf_341, buf_611, buf_612, buf_343], [4, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[553], layouts[553], infinityBuf, [buf_168, buf_384, buf_613, buf_614], [2, 320, 2]);
        addComputePass(device, commandEncoder, pipelines[554], layouts[554], infinityBuf, [buf_615, buf_168, buf_249], [16, 240, 2]);
        addComputePass(device, commandEncoder, pipelines[555], layouts[555], infinityBuf, [buf_76, buf_615], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[556], layouts[556], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[557], layouts[557], infinityBuf, [buf_263, buf_615, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[558], layouts[558], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[559], layouts[559], infinityBuf, [buf_616, buf_615, buf_80, buf_148, buf_617, buf_618], [16, 240, 2]);
        addComputePass(device, commandEncoder, pipelines[560], layouts[560], infinityBuf, [buf_249, buf_616, buf_619, buf_620, buf_130], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[561], layouts[561], infinityBuf, [buf_80, buf_249], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[562], layouts[562], infinityBuf, [buf_148, buf_249, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[563], layouts[563], infinityBuf, [buf_256, buf_249, buf_80, buf_148, buf_621, buf_622], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[564], layouts[564], infinityBuf, [buf_242, buf_256, buf_623], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[565], layouts[565], infinityBuf, [buf_299, buf_615, buf_624, buf_625, buf_242, buf_626], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[566], layouts[566], infinityBuf, [buf_80, buf_299], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[567], layouts[567], infinityBuf, [buf_148, buf_299, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[568], layouts[568], infinityBuf, [buf_260, buf_299, buf_80, buf_148, buf_627, buf_628], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[569], layouts[569], infinityBuf, [buf_239, buf_260, buf_629, buf_630], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[570], layouts[570], infinityBuf, [buf_252, buf_239], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[571], layouts[571], infinityBuf, [buf_253, buf_239, buf_252], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[572], layouts[572], infinityBuf, [buf_236, buf_239, buf_252, buf_253, buf_631, buf_632], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[573], layouts[573], infinityBuf, [buf_249, buf_236, buf_633], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[574], layouts[574], infinityBuf, [buf_256, buf_236, buf_634], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[575], layouts[575], infinityBuf, [buf_260, buf_236, buf_635], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[576], layouts[576], infinityBuf, [buf_262, buf_249, buf_256], [512, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[577], layouts[577], infinityBuf, [buf_76, buf_262], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[578], layouts[578], infinityBuf, [buf_263, buf_262, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[579], layouts[579], infinityBuf, [buf_264, buf_262, buf_76, buf_263], [16, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[580], layouts[580], infinityBuf, [buf_236, buf_264, buf_260], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[581], layouts[581], infinityBuf, [buf_249, buf_239, buf_236, buf_636, buf_637], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[582], layouts[582], infinityBuf, [buf_252, buf_249], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[583], layouts[583], infinityBuf, [buf_253, buf_249, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[584], layouts[584], infinityBuf, [buf_239, buf_249, buf_252, buf_253, buf_638, buf_639], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[585], layouts[585], infinityBuf, [buf_256, buf_239, buf_640], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[586], layouts[586], infinityBuf, [buf_270, buf_256, buf_52], [77, 16, 2]);
        addComputePass(device, commandEncoder, pipelines[587], layouts[587], infinityBuf, [buf_76, buf_270], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[588], layouts[588], infinityBuf, [buf_263, buf_270, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[589], layouts[589], infinityBuf, [buf_271, buf_270, buf_76, buf_263], [77, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[590], layouts[590], infinityBuf, [buf_260, buf_271, buf_54], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[591], layouts[591], infinityBuf, [buf_236, buf_249, buf_260, buf_641, buf_642], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[592], layouts[592], infinityBuf, [buf_252, buf_236], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[593], layouts[593], infinityBuf, [buf_253, buf_236, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[594], layouts[594], infinityBuf, [buf_249, buf_236, buf_252, buf_253, buf_643, buf_644], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[595], layouts[595], infinityBuf, [buf_192, buf_249, buf_645, buf_646], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[596], layouts[596], infinityBuf, [buf_278, buf_192], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[597], layouts[597], infinityBuf, [buf_239, buf_236, buf_278, buf_647, buf_648], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[598], layouts[598], infinityBuf, [buf_256, buf_239, buf_649, buf_650, buf_299], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[599], layouts[599], infinityBuf, [buf_213, buf_256, buf_258], [16, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[600], layouts[600], infinityBuf, [buf_76, buf_213], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[601], layouts[601], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[602], layouts[602], infinityBuf, [buf_263, buf_213, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[603], layouts[603], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[604], layouts[604], infinityBuf, [buf_172, buf_213, buf_80, buf_148, buf_651, buf_652], [16, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[605], layouts[605], infinityBuf, [buf_258, buf_172, buf_653, buf_654, buf_133], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[606], layouts[606], infinityBuf, [buf_80, buf_258], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[607], layouts[607], infinityBuf, [buf_148, buf_258, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[608], layouts[608], infinityBuf, [buf_299, buf_258, buf_80, buf_148, buf_655, buf_656], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[609], layouts[609], infinityBuf, [buf_242, buf_299, buf_657], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[610], layouts[610], infinityBuf, [buf_260, buf_213, buf_658, buf_659, buf_242, buf_660], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[611], layouts[611], infinityBuf, [buf_80, buf_260], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[612], layouts[612], infinityBuf, [buf_148, buf_260, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[613], layouts[613], infinityBuf, [buf_236, buf_260, buf_80, buf_148, buf_661, buf_662], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[614], layouts[614], infinityBuf, [buf_249, buf_236, buf_663, buf_664], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[615], layouts[615], infinityBuf, [buf_252, buf_249], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[616], layouts[616], infinityBuf, [buf_253, buf_249, buf_252], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[617], layouts[617], infinityBuf, [buf_239, buf_249, buf_252, buf_253, buf_665, buf_666], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[618], layouts[618], infinityBuf, [buf_256, buf_239, buf_667], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[619], layouts[619], infinityBuf, [buf_258, buf_239, buf_668], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[620], layouts[620], infinityBuf, [buf_299, buf_239, buf_669], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[621], layouts[621], infinityBuf, [buf_262, buf_256, buf_258], [512, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[622], layouts[622], infinityBuf, [buf_76, buf_262], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[623], layouts[623], infinityBuf, [buf_263, buf_262, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[624], layouts[624], infinityBuf, [buf_264, buf_262, buf_76, buf_263], [16, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[625], layouts[625], infinityBuf, [buf_236, buf_264, buf_299], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[626], layouts[626], infinityBuf, [buf_239, buf_249, buf_236, buf_670, buf_671], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[627], layouts[627], infinityBuf, [buf_252, buf_239], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[628], layouts[628], infinityBuf, [buf_253, buf_239, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[629], layouts[629], infinityBuf, [buf_249, buf_239, buf_252, buf_253, buf_672, buf_673], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[630], layouts[630], infinityBuf, [buf_256, buf_249, buf_674], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[631], layouts[631], infinityBuf, [buf_270, buf_256, buf_56], [77, 16, 2]);
        addComputePass(device, commandEncoder, pipelines[632], layouts[632], infinityBuf, [buf_76, buf_270], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[633], layouts[633], infinityBuf, [buf_263, buf_270, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[634], layouts[634], infinityBuf, [buf_271, buf_270, buf_76, buf_263], [77, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[635], layouts[635], infinityBuf, [buf_258, buf_271, buf_58], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[636], layouts[636], infinityBuf, [buf_299, buf_239, buf_258, buf_675, buf_676], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[637], layouts[637], infinityBuf, [buf_252, buf_299], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[638], layouts[638], infinityBuf, [buf_253, buf_299, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[639], layouts[639], infinityBuf, [buf_236, buf_299, buf_252, buf_253, buf_677, buf_678], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[640], layouts[640], infinityBuf, [buf_192, buf_236, buf_679, buf_680], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[641], layouts[641], infinityBuf, [buf_278, buf_192], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[642], layouts[642], infinityBuf, [buf_239, buf_299, buf_278, buf_681, buf_682], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[643], layouts[643], infinityBuf, [buf_249, buf_239, buf_683, buf_684, buf_260], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[644], layouts[644], infinityBuf, [buf_685, buf_249, buf_230], [16, 120, 2]);
        addComputePass(device, commandEncoder, pipelines[645], layouts[645], infinityBuf, [buf_80, buf_685], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[646], layouts[646], infinityBuf, [buf_148, buf_685, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[647], layouts[647], infinityBuf, [buf_686, buf_685, buf_80, buf_148, buf_687, buf_688], [16, 120, 2]);
        addComputePass(device, commandEncoder, pipelines[648], layouts[648], infinityBuf, [buf_260, buf_686, buf_689, buf_690, buf_136], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[649], layouts[649], infinityBuf, [buf_80, buf_260], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[650], layouts[650], infinityBuf, [buf_148, buf_260, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[651], layouts[651], infinityBuf, [buf_256, buf_260, buf_80, buf_148, buf_691, buf_692], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[652], layouts[652], infinityBuf, [buf_242, buf_256, buf_693], [2, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[653], layouts[653], infinityBuf, [buf_258, buf_685, buf_694, buf_695, buf_242, buf_696], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[654], layouts[654], infinityBuf, [buf_80, buf_258], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[655], layouts[655], infinityBuf, [buf_148, buf_258, buf_80], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[656], layouts[656], infinityBuf, [buf_299, buf_258, buf_80, buf_148, buf_697, buf_698], [16, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[657], layouts[657], infinityBuf, [buf_236, buf_299, buf_699, buf_700], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[658], layouts[658], infinityBuf, [buf_252, buf_236], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[659], layouts[659], infinityBuf, [buf_253, buf_236, buf_252], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[660], layouts[660], infinityBuf, [buf_239, buf_236, buf_252, buf_253, buf_701, buf_702], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[661], layouts[661], infinityBuf, [buf_249, buf_239, buf_703], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[662], layouts[662], infinityBuf, [buf_260, buf_239, buf_704], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[663], layouts[663], infinityBuf, [buf_256, buf_239, buf_705], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[664], layouts[664], infinityBuf, [buf_262, buf_249, buf_260], [512, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[665], layouts[665], infinityBuf, [buf_76, buf_262], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[666], layouts[666], infinityBuf, [buf_263, buf_262, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[667], layouts[667], infinityBuf, [buf_264, buf_262, buf_76, buf_263], [16, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[668], layouts[668], infinityBuf, [buf_299, buf_264, buf_256], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[669], layouts[669], infinityBuf, [buf_239, buf_236, buf_299, buf_706, buf_707], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[670], layouts[670], infinityBuf, [buf_252, buf_239], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[671], layouts[671], infinityBuf, [buf_253, buf_239, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[672], layouts[672], infinityBuf, [buf_236, buf_239, buf_252, buf_253, buf_708, buf_709], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[673], layouts[673], infinityBuf, [buf_249, buf_236, buf_710], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[674], layouts[674], infinityBuf, [buf_270, buf_249, buf_60], [77, 16, 2]);
        addComputePass(device, commandEncoder, pipelines[675], layouts[675], infinityBuf, [buf_76, buf_270], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[676], layouts[676], infinityBuf, [buf_263, buf_270, buf_76], [128, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[677], layouts[677], infinityBuf, [buf_271, buf_270, buf_76, buf_263], [77, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[678], layouts[678], infinityBuf, [buf_260, buf_271, buf_62], [80, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[679], layouts[679], infinityBuf, [buf_256, buf_239, buf_260, buf_711, buf_712], [10, 32, 2]);
        addComputePass(device, commandEncoder, pipelines[680], layouts[680], infinityBuf, [buf_252, buf_256], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[681], layouts[681], infinityBuf, [buf_253, buf_256, buf_252], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[682], layouts[682], infinityBuf, [buf_299, buf_256, buf_252, buf_253, buf_713, buf_714], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[683], layouts[683], infinityBuf, [buf_192, buf_299, buf_715, buf_716], [80, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[684], layouts[684], infinityBuf, [buf_278, buf_192], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[685], layouts[685], infinityBuf, [buf_239, buf_256, buf_278, buf_717, buf_718], [10, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[686], layouts[686], infinityBuf, [buf_236, buf_239, buf_719, buf_720, buf_258], [16, 20, 2]);
        addComputePass(device, commandEncoder, pipelines[687], layouts[687], infinityBuf, [buf_278, buf_236, buf_721, buf_722], [8, 160, 2]);
        addComputePass(device, commandEncoder, pipelines[688], layouts[688], infinityBuf, [buf_723, buf_278, buf_161], [64, 120, 2]);
        addComputePass(device, commandEncoder, pipelines[689], layouts[689], infinityBuf, [buf_76, buf_723], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[690], layouts[690], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[691], layouts[691], infinityBuf, [buf_263, buf_723, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[692], layouts[692], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[693], layouts[693], infinityBuf, [buf_724, buf_723, buf_80, buf_148, buf_725, buf_726], [64, 120, 2]);
        addComputePass(device, commandEncoder, pipelines[694], layouts[694], infinityBuf, [buf_161, buf_724, buf_727, buf_728, buf_139], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[695], layouts[695], infinityBuf, [buf_76, buf_161], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[696], layouts[696], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[697], layouts[697], infinityBuf, [buf_263, buf_161, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[698], layouts[698], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[699], layouts[699], infinityBuf, [buf_149, buf_161, buf_80, buf_148, buf_729, buf_730], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[700], layouts[700], infinityBuf, [buf_731, buf_149, buf_732], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[701], layouts[701], infinityBuf, [buf_152, buf_723, buf_733, buf_734, buf_731, buf_735], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[702], layouts[702], infinityBuf, [buf_76, buf_152], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[703], layouts[703], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[704], layouts[704], infinityBuf, [buf_263, buf_152, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[705], layouts[705], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[706], layouts[706], infinityBuf, [buf_168, buf_152, buf_80, buf_148, buf_736, buf_737], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[707], layouts[707], infinityBuf, [buf_213, buf_168, buf_738, buf_739], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[708], layouts[708], infinityBuf, [buf_164, buf_213], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[709], layouts[709], infinityBuf, [buf_165, buf_213, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[710], layouts[710], infinityBuf, [buf_172, buf_213, buf_164, buf_165, buf_740, buf_741], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[711], layouts[711], infinityBuf, [buf_161, buf_172, buf_742], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[712], layouts[712], infinityBuf, [buf_149, buf_172, buf_743], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[713], layouts[713], infinityBuf, [buf_168, buf_172, buf_744], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[714], layouts[714], infinityBuf, [buf_174, buf_161, buf_149], [8192, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[715], layouts[715], infinityBuf, [buf_176, buf_174], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[716], layouts[716], infinityBuf, [buf_175, buf_174, buf_176], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[717], layouts[717], infinityBuf, [buf_177, buf_174, buf_176, buf_175], [64, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[718], layouts[718], infinityBuf, [buf_172, buf_177, buf_168], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[719], layouts[719], infinityBuf, [buf_161, buf_213, buf_172, buf_745, buf_746], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[720], layouts[720], infinityBuf, [buf_164, buf_161], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[721], layouts[721], infinityBuf, [buf_165, buf_161, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[722], layouts[722], infinityBuf, [buf_213, buf_161, buf_164, buf_165, buf_747, buf_748], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[723], layouts[723], infinityBuf, [buf_149, buf_213, buf_749], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[724], layouts[724], infinityBuf, [buf_183, buf_149, buf_64], [77, 64, 2]);
        addComputePass(device, commandEncoder, pipelines[725], layouts[725], infinityBuf, [buf_176, buf_183], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[726], layouts[726], infinityBuf, [buf_175, buf_183, buf_176], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[727], layouts[727], infinityBuf, [buf_184, buf_183, buf_176, buf_175], [77, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[728], layouts[728], infinityBuf, [buf_168, buf_184, buf_66], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[729], layouts[729], infinityBuf, [buf_172, buf_161, buf_168, buf_750, buf_751], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[730], layouts[730], infinityBuf, [buf_164, buf_172], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[731], layouts[731], infinityBuf, [buf_165, buf_172, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[732], layouts[732], infinityBuf, [buf_161, buf_172, buf_164, buf_165, buf_752, buf_753], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[733], layouts[733], infinityBuf, [buf_189, buf_161, buf_754, buf_755], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[734], layouts[734], infinityBuf, [buf_192, buf_189], [20, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[735], layouts[735], infinityBuf, [buf_213, buf_172, buf_192, buf_756, buf_757], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[736], layouts[736], infinityBuf, [buf_149, buf_213, buf_758, buf_759, buf_152], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[737], layouts[737], infinityBuf, [buf_278, buf_149, buf_170], [64, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[738], layouts[738], infinityBuf, [buf_76, buf_278], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[739], layouts[739], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[740], layouts[740], infinityBuf, [buf_263, buf_278, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[741], layouts[741], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[742], layouts[742], infinityBuf, [buf_760, buf_278, buf_80, buf_148, buf_761, buf_762], [64, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[743], layouts[743], infinityBuf, [buf_170, buf_760, buf_763, buf_764, buf_142], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[744], layouts[744], infinityBuf, [buf_76, buf_170], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[745], layouts[745], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[746], layouts[746], infinityBuf, [buf_263, buf_170, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[747], layouts[747], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[748], layouts[748], infinityBuf, [buf_152, buf_170, buf_80, buf_148, buf_765, buf_766], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[749], layouts[749], infinityBuf, [buf_731, buf_152, buf_767], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[750], layouts[750], infinityBuf, [buf_168, buf_278, buf_768, buf_769, buf_731, buf_770], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[751], layouts[751], infinityBuf, [buf_76, buf_168], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[752], layouts[752], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[753], layouts[753], infinityBuf, [buf_263, buf_168, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[754], layouts[754], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[755], layouts[755], infinityBuf, [buf_172, buf_168, buf_80, buf_148, buf_771, buf_772], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[756], layouts[756], infinityBuf, [buf_161, buf_172, buf_773, buf_774], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[757], layouts[757], infinityBuf, [buf_164, buf_161], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[758], layouts[758], infinityBuf, [buf_165, buf_161, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[759], layouts[759], infinityBuf, [buf_213, buf_161, buf_164, buf_165, buf_775, buf_776], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[760], layouts[760], infinityBuf, [buf_149, buf_213, buf_777], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[761], layouts[761], infinityBuf, [buf_170, buf_213, buf_778], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[762], layouts[762], infinityBuf, [buf_152, buf_213, buf_779], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[763], layouts[763], infinityBuf, [buf_174, buf_149, buf_170], [8192, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[764], layouts[764], infinityBuf, [buf_176, buf_174], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[765], layouts[765], infinityBuf, [buf_175, buf_174, buf_176], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[766], layouts[766], infinityBuf, [buf_177, buf_174, buf_176, buf_175], [64, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[767], layouts[767], infinityBuf, [buf_172, buf_177, buf_152], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[768], layouts[768], infinityBuf, [buf_213, buf_161, buf_172, buf_780, buf_781], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[769], layouts[769], infinityBuf, [buf_164, buf_213], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[770], layouts[770], infinityBuf, [buf_165, buf_213, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[771], layouts[771], infinityBuf, [buf_161, buf_213, buf_164, buf_165, buf_782, buf_783], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[772], layouts[772], infinityBuf, [buf_149, buf_161, buf_784], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[773], layouts[773], infinityBuf, [buf_183, buf_149, buf_68], [77, 64, 2]);
        addComputePass(device, commandEncoder, pipelines[774], layouts[774], infinityBuf, [buf_176, buf_183], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[775], layouts[775], infinityBuf, [buf_175, buf_183, buf_176], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[776], layouts[776], infinityBuf, [buf_184, buf_183, buf_176, buf_175], [77, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[777], layouts[777], infinityBuf, [buf_170, buf_184, buf_70], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[778], layouts[778], infinityBuf, [buf_152, buf_213, buf_170, buf_785, buf_786], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[779], layouts[779], infinityBuf, [buf_164, buf_152], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[780], layouts[780], infinityBuf, [buf_165, buf_152, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[781], layouts[781], infinityBuf, [buf_172, buf_152, buf_164, buf_165, buf_787, buf_788], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[782], layouts[782], infinityBuf, [buf_189, buf_172, buf_789, buf_790], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[783], layouts[783], infinityBuf, [buf_192, buf_189], [20, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[784], layouts[784], infinityBuf, [buf_213, buf_152, buf_192, buf_791, buf_792], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[785], layouts[785], infinityBuf, [buf_161, buf_213, buf_793, buf_794, buf_168], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[786], layouts[786], infinityBuf, [buf_278, buf_161, buf_8], [64, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[787], layouts[787], infinityBuf, [buf_76, buf_278], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[788], layouts[788], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[789], layouts[789], infinityBuf, [buf_263, buf_278, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[790], layouts[790], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[791], layouts[791], infinityBuf, [buf_760, buf_278, buf_80, buf_148, buf_795, buf_796], [64, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[792], layouts[792], infinityBuf, [buf_8, buf_760, buf_797, buf_798, buf_145], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[793], layouts[793], infinityBuf, [buf_76, buf_8], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[794], layouts[794], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[795], layouts[795], infinityBuf, [buf_263, buf_8, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[796], layouts[796], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[797], layouts[797], infinityBuf, [buf_168, buf_8, buf_80, buf_148, buf_799, buf_800], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[798], layouts[798], infinityBuf, [buf_731, buf_168, buf_801], [8, 80, 2]);
        addComputePass(device, commandEncoder, pipelines[799], layouts[799], infinityBuf, [buf_149, buf_278, buf_802, buf_803, buf_731, buf_804], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[800], layouts[800], infinityBuf, [buf_76, buf_149], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[801], layouts[801], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[802], layouts[802], infinityBuf, [buf_263, buf_149, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[803], layouts[803], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[804], layouts[804], infinityBuf, [buf_170, buf_149, buf_80, buf_148, buf_805, buf_806], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[805], layouts[805], infinityBuf, [buf_152, buf_170, buf_807, buf_808], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[806], layouts[806], infinityBuf, [buf_164, buf_152], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[807], layouts[807], infinityBuf, [buf_165, buf_152, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[808], layouts[808], infinityBuf, [buf_172, buf_152, buf_164, buf_165, buf_809, buf_810], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[809], layouts[809], infinityBuf, [buf_213, buf_172, buf_811], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[810], layouts[810], infinityBuf, [buf_161, buf_172, buf_812], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[811], layouts[811], infinityBuf, [buf_8, buf_172, buf_813], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[812], layouts[812], infinityBuf, [buf_174, buf_213, buf_161], [8192, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[813], layouts[813], infinityBuf, [buf_176, buf_174], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[814], layouts[814], infinityBuf, [buf_175, buf_174, buf_176], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[815], layouts[815], infinityBuf, [buf_177, buf_174, buf_176, buf_175], [64, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[816], layouts[816], infinityBuf, [buf_168, buf_177, buf_8], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[817], layouts[817], infinityBuf, [buf_170, buf_152, buf_168, buf_814, buf_815], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[818], layouts[818], infinityBuf, [buf_164, buf_170], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[819], layouts[819], infinityBuf, [buf_165, buf_170, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[820], layouts[820], infinityBuf, [buf_152, buf_170, buf_164, buf_165, buf_816, buf_817], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[821], layouts[821], infinityBuf, [buf_172, buf_152, buf_818], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[822], layouts[822], infinityBuf, [buf_183, buf_172, buf_72], [77, 64, 2]);
        addComputePass(device, commandEncoder, pipelines[823], layouts[823], infinityBuf, [buf_176, buf_183], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[824], layouts[824], infinityBuf, [buf_175, buf_183, buf_176], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[825], layouts[825], infinityBuf, [buf_184, buf_183, buf_176, buf_175], [77, 512, 1]);
        addComputePass(device, commandEncoder, pipelines[826], layouts[826], infinityBuf, [buf_213, buf_184, buf_74], [320, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[827], layouts[827], infinityBuf, [buf_161, buf_170, buf_213, buf_819, buf_820], [5, 128, 2]);
        addComputePass(device, commandEncoder, pipelines[828], layouts[828], infinityBuf, [buf_164, buf_161], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[829], layouts[829], infinityBuf, [buf_165, buf_161, buf_164], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[830], layouts[830], infinityBuf, [buf_8, buf_161, buf_164, buf_165, buf_821, buf_822], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[831], layouts[831], infinityBuf, [buf_189, buf_8, buf_823, buf_824], [40, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[832], layouts[832], infinityBuf, [buf_192, buf_189], [20, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[833], layouts[833], infinityBuf, [buf_168, buf_161, buf_192, buf_825, buf_826], [5, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[834], layouts[834], infinityBuf, [buf_170, buf_168, buf_827, buf_828, buf_149], [64, 10, 2]);
        addComputePass(device, commandEncoder, pipelines[835], layouts[835], infinityBuf, [buf_76, buf_170], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[836], layouts[836], infinityBuf, [buf_80, buf_76], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[837], layouts[837], infinityBuf, [buf_263, buf_170, buf_80], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[838], layouts[838], infinityBuf, [buf_148, buf_263], [64, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[839], layouts[839], infinityBuf, [buf_149, buf_170, buf_80, buf_148, buf_829, buf_830], [64, 40, 2]);
        addComputePass(device, commandEncoder, pipelines[840], layouts[840], infinityBuf, [buf_831, buf_149, buf_832, buf_833], [8, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[841], layouts[841], infinityBuf, [output0, buf_0, input2, buf_1, buf_831, input6, buf_6, buf_7], [128, 1, 1]);
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
return { load, getWeight };
})();
export default diffusor;