// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright 2026 Parham (Leon) Faraji (@leon-executes).
// Optional, isolated OpenCL executor. Uses the installed GPU driver; no SDK
// runtime bundled.
#define WIN32_LEAN_AND_MEAN
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
using I = int;
using U = unsigned;
using B = uint64_t;
using H = void *;
using Z = size_t;
#define API(name, ret, args)                                                   \
  using name##T = ret(WINAPI *) args;                                          \
  name##T name = nullptr
API(clGetPlatformIDs, I, (U, H *, U *));
API(clGetDeviceIDs, I, (H, B, U, H *, U *));
API(clGetDeviceInfo, I, (H, U, Z, void *, Z *));
API(clCreateContext, H, (const intptr_t *, U, const H *, void *, void *, I *));
API(clCreateCommandQueue, H, (H, H, B, I *));
API(clCreateProgramWithSource, H, (H, U, const char **, const Z *, I *));
API(clBuildProgram, I, (H, U, const H *, const char *, void *, void *));
API(clGetProgramBuildInfo, I, (H, H, U, Z, void *, Z *));
API(clCreateKernel, H, (H, const char *, I *));
API(clCreateBuffer, H, (H, B, Z, void *, I *));
API(clSetKernelArg, I, (H, U, Z, const void *));
API(clEnqueueNDRangeKernel, I,
    (H, H, U, const Z *, const Z *, const Z *, U, const H *, H *));
API(clEnqueueReadBuffer, I, (H, H, U, Z, Z, void *, U, const H *, H *));
API(clFinish, I, (H));
API(clReleaseMemObject, I, (H));
API(clReleaseKernel, I, (H));
API(clReleaseProgram, I, (H));
API(clReleaseCommandQueue, I, (H));
API(clReleaseContext, I, (H));
static void check(I e) {
  if (e)
    throw std::runtime_error("OpenCL error " + std::to_string(e));
}
static void load() {
  HMODULE lib =
      LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!lib)
    throw std::runtime_error("No OpenCL driver runtime");
#define LOAD(n)                                                                \
  n = (n##T)GetProcAddress(lib, #n);                                           \
  if (!n)                                                                      \
  throw std::runtime_error("Missing " #n)
  LOAD(clGetPlatformIDs);
  LOAD(clGetDeviceIDs);
  LOAD(clGetDeviceInfo);
  LOAD(clCreateContext);
  LOAD(clCreateCommandQueue);
  LOAD(clCreateProgramWithSource);
  LOAD(clBuildProgram);
  LOAD(clGetProgramBuildInfo);
  LOAD(clCreateKernel);
  LOAD(clCreateBuffer);
  LOAD(clSetKernelArg);
  LOAD(clEnqueueNDRangeKernel);
  LOAD(clEnqueueReadBuffer);
  LOAD(clFinish);
  LOAD(clReleaseMemObject);
  LOAD(clReleaseKernel);
  LOAD(clReleaseProgram);
  LOAD(clReleaseCommandQueue);
  LOAD(clReleaseContext);
}
template <class T> T info(H d, U key) {
  T v{};
  check(clGetDeviceInfo(d, key, sizeof(v), &v, nullptr));
  return v;
}
static std::string text(H d, U key) {
  char s[8192] = {};
  check(clGetDeviceInfo(d, key, sizeof(s), s, nullptr));
  return s;
}
static std::string json(const std::string &s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      o += '\\';
    if (c >= 32)
      o += c;
  }
  return o + '"';
}
struct Device {
  H id;
  std::string name, driver;
  B memory, allocation;
  U units, clock, unified;
  double rank;
};
static std::vector<Device> devices() {
  U n = 0;
  check(clGetPlatformIDs(0, nullptr, &n));
  std::vector<H> p(n);
  check(clGetPlatformIDs(n, p.data(), nullptr));
  std::vector<Device> out;
  for (H platform : p) {
    U count = 0;
    if (clGetDeviceIDs(platform, 4, 0, nullptr, &count))
      continue;
    std::vector<H> d(count);
    check(clGetDeviceIDs(platform, 4, count, d.data(), nullptr));
    for (H id : d) {
      if (!info<U>(id, 0x1027) ||
          text(id, 0x1030).find("cl_khr_fp64") == std::string::npos)
        continue;
      Device v{id,
               text(id, 0x102b),
               text(id, 0x102d),
               info<B>(id, 0x101f),
               info<B>(id, 0x1010),
               info<U>(id, 0x1002),
               info<U>(id, 0x100c),
               info<U>(id, 0x1035),
               0};
      v.rank = double(v.units) * v.clock * (v.unified ? 1 : 2);
      out.push_back(v);
    }
  }
  std::stable_sort(
      out.begin(), out.end(),
      [](const Device &a, const Device &b) { return a.rank > b.rank; });
  return out;
}
static const char *source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
constant double weights[5]={.05,.25,.4,.25,.05};
kernel void remap_field(global const double*a,global double*m,int w,int h,int first,int count,int bins){
 int i=get_global_id(0),n=w*h;if(i>=n*count)return;int refi=first+i/n;double ref=(double)refi/(bins-1),v=a[i%n],d=fabs(v-ref),z=d/.075;
 double t=clamp((d-.01)/.01,0.0,1.0);t=t*t*(t-2)*(t-2);double mapped=t*pow(z,.65)+(1-t)*z;
 m[i]=ref+(v>ref?1.0:v<ref?-1.0:0.0)*(d<.075?.075*mapped:d);
}
kernel void reduce_field(global const double*a,global double*b,int w,int h,int nw,int nh,int count){
 int i=get_global_id(0),n=nw*nh;if(i>=n*count)return;int z=i/n,x=2*(i%n%nw),y=2*(i%n/nw);double v=0,norm=0;
 for(int ky=-2;ky<=2;ky++){if(y+ky<0||y+ky>=h)continue;double row=0,ws=0;
  for(int kx=-2;kx<=2;kx++){if(x+kx<0||x+kx>=w)continue;row+=weights[kx+2]*a[z*w*h+(y+ky)*w+x+kx];ws+=weights[kx+2];}
  v+=weights[ky+2]*row;norm+=weights[ky+2]*ws;
 }b[i]=v/norm;
}
double expand(global const double*a,int x,int y,int w,int h,int sw,int sh,int z){
 double v=0,norm=0;
 for(int ky=-2;ky<=2;ky++){int yy=y+ky;if(yy<0||yy>=h||(yy&1))continue;double row=0,ws=0;
  for(int kx=-2;kx<=2;kx++){int xx=x+kx;if(xx<0||xx>=w||(xx&1))continue;row+=weights[kx+2]*a[z*sw*sh+(yy/2)*sw+xx/2];ws+=weights[kx+2];}
  v+=weights[ky+2]*row;norm+=weights[ky+2]*ws;
 }return v/norm;
}
kernel void accumulate(global const double*g,global const double*cur,global const double*next,global double*out,int w,int h,int sw,int sh,int first,int count,int bins){
 int i=get_global_id(0),n=w*h;if(i>=n)return;int x=i%w,y=i/w;double v=out[i];
 for(int z=0;z<count;z++){double ref=(double)(first+z)/(bins-1),l=cur[z*n+i]-expand(next,x,y,w,h,sw,sh,z);v+=l*fmax(0.0,1-fabs(g[i]-ref)*(bins-1));}out[i]=v;
}
kernel void reconstruct(global const double*a,global const double*b,global double*out,int w,int h,int sw,int sh){int i=get_global_id(0);if(i>=w*h)return;out[i]=a[i]+expand(b,i%w,i/w,w,h,sw,sh,0);}
)CLC";
struct Compute {
  H context = nullptr, queue = nullptr, program = nullptr;
  std::vector<H> buffers, kernels;
  Device device;
  Compute(const Device &d) : device(d) {
    I e = 0;
    context = clCreateContext(nullptr, 1, &device.id, nullptr, nullptr, &e);
    check(e);
    queue = clCreateCommandQueue(context, device.id, 0, &e);
    check(e);
    const char *s = source;
    program = clCreateProgramWithSource(context, 1, &s, nullptr, &e);
    check(e);
    e = clBuildProgram(program, 1, &device.id, "-cl-std=CL1.2", nullptr,
                       nullptr);
    if (e) {
      char log[8192] = {};
      clGetProgramBuildInfo(program, device.id, 0x1183, sizeof(log), log,
                            nullptr);
      throw std::runtime_error(log);
    }
  }
  ~Compute() {
    for (H k : kernels)
      clReleaseKernel(k);
    for (H b : buffers)
      clReleaseMemObject(b);
    if (program)
      clReleaseProgram(program);
    if (queue)
      clReleaseCommandQueue(queue);
    if (context)
      clReleaseContext(context);
  }
  H buffer(Z n, const double *p = nullptr) {
    if (n > device.allocation || n > device.memory / 3)
      throw std::runtime_error("GPU allocation limit");
    I e;
    H b = clCreateBuffer(context, 1 | (p ? 32 : 0), n, (void *)p, &e);
    check(e);
    buffers.push_back(b);
    return b;
  }
  H kernel(const char *n) {
    I e;
    H k = clCreateKernel(program, n, &e);
    check(e);
    kernels.push_back(k);
    return k;
  }
  template <class T> void arg(H k, U i, T v) {
    check(clSetKernelArg(k, i, sizeof(v), &v));
  }
  void run(H k, Z n) {
    check(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &n, nullptr, 0, nullptr,
                                 nullptr));
  }
  std::vector<double> laplacian(const std::vector<double> &input, int w, int h,
                                int bins) {
    const int batch = 8;
    std::vector<int> ws{w}, hs{h};
    while (ws.size() < 6 && std::min(ws.back(), hs.back()) > 2) {
      ws.push_back((ws.back() + 1) / 2);
      hs.push_back((hs.back() + 1) / 2);
    }
    std::vector<H> g, m, o, r;
    std::vector<double> zero(Z(w) * h, 0);
    int levels = int(ws.size());
    for (int l = 0; l < levels; l++) {
      Z bytes = Z(ws[l]) * hs[l] * 8;
      g.push_back(buffer(bytes, l == 0 ? input.data() : nullptr));
      m.push_back(buffer(bytes * batch));
      o.push_back(buffer(bytes, zero.data()));
      r.push_back(buffer(bytes));
    }
    H reduce = kernel("reduce_field"), remap = kernel("remap_field"),
      acc = kernel("accumulate"), collapse = kernel("reconstruct");
    auto down = [&](H a, H b, int l, int count) {
      arg(reduce, 0, a);
      arg(reduce, 1, b);
      arg(reduce, 2, ws[l]);
      arg(reduce, 3, hs[l]);
      arg(reduce, 4, ws[l + 1]);
      arg(reduce, 5, hs[l + 1]);
      arg(reduce, 6, count);
      run(reduce, Z(ws[l + 1]) * hs[l + 1] * count);
    };
    for (int l = 0; l < levels - 1; l++)
      down(g[l], g[l + 1], l, 1);
    for (int first = 0; first < bins; first += batch) {
      int count = std::min(batch, bins - first);
      arg(remap, 0, g[0]);
      arg(remap, 1, m[0]);
      arg(remap, 2, w);
      arg(remap, 3, h);
      arg(remap, 4, first);
      arg(remap, 5, count);
      arg(remap, 6, bins);
      run(remap, Z(w) * h * count);
      for (int l = 0; l < levels - 1; l++) {
        down(m[l], m[l + 1], l, count);
        arg(acc, 0, g[l]);
        arg(acc, 1, m[l]);
        arg(acc, 2, m[l + 1]);
        arg(acc, 3, o[l]);
        arg(acc, 4, ws[l]);
        arg(acc, 5, hs[l]);
        arg(acc, 6, ws[l + 1]);
        arg(acc, 7, hs[l + 1]);
        arg(acc, 8, first);
        arg(acc, 9, count);
        arg(acc, 10, bins);
        run(acc, Z(ws[l]) * hs[l]);
      }
    }
    H last = g.back();
    for (int l = levels - 2; l >= 0; l--) {
      arg(collapse, 0, o[l]);
      arg(collapse, 1, last);
      arg(collapse, 2, r[l]);
      arg(collapse, 3, ws[l]);
      arg(collapse, 4, hs[l]);
      arg(collapse, 5, ws[l + 1]);
      arg(collapse, 6, hs[l + 1]);
      run(collapse, Z(ws[l]) * hs[l]);
      last = r[l];
    }
    std::vector<double> result(Z(w) * h);
    check(clEnqueueReadBuffer(queue, last, 1, 0, result.size() * 8,
                              result.data(), 0, nullptr, nullptr));
    check(clFinish(queue));
    return result;
  }
};
int main(int argc, char **argv) {
  try {
    load();
    auto ds = devices();
    if (argc == 2 && std::string(argv[1]) == "--probe") {
      std::cout << "[";
      for (Z i = 0; i < ds.size(); i++) {
        auto &d = ds[i];
        if (i)
          std::cout << ",";
        std::cout << "{\"index\":" << i << ",\"name\":" << json(d.name)
                  << ",\"driver\":" << json(d.driver)
                  << ",\"memory\":" << d.memory << ",\"unified\":" << d.unified
                  << "}";
      }
      std::cout << "]\n";
      return 0;
    }
    if (argc != 6)
      throw std::runtime_error(
          "Usage: untorn_gpu input.f64 output.f64 bins device_index warm_runs");
    int bins = atoi(argv[3]), index = atoi(argv[4]), runs = atoi(argv[5]);
    if (bins < 2 || bins > 80 || index < 0 || index >= int(ds.size()) ||
        runs < 1 || runs > 10)
      throw std::runtime_error("Invalid settings");
    std::ifstream f(argv[1], std::ios::binary);
    uint32_t wh[2] = {};
    f.read((char *)wh, 8);
    int w = wh[0], h = wh[1];
    if (w < 2 || h < 2 || w > 1024 || h > 1024)
      throw std::runtime_error("Image dimensions out of range");
    std::vector<double> input(Z(w) * h);
    f.read((char *)input.data(), input.size() * 8);
    if (!f)
      throw std::runtime_error("Truncated input");
    for (double &x : input) {
      if (!std::isfinite(x) || x < 0 || x > 255)
        throw std::runtime_error("Invalid input values");
      x /= 255;
    }
    auto start = std::chrono::steady_clock::now();
    Compute c(ds[index]);
    std::vector<double> result;
    for (int i = 0; i < runs; i++)
      result = c.laplacian(input, w, h, bins);
    for (double &x : result) {
      x *= 255;
      if (!std::isfinite(x))
        throw std::runtime_error("Non-finite GPU output");
    }
    std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
    out.write((char *)wh, 8);
    out.write((char *)result.data(), result.size() * 8);
    out.close();
    if (!out)
      throw std::runtime_error("Output write failed");
    std::cout << "{\"device\":" << json(ds[index].name) << ",\"seconds\":"
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
