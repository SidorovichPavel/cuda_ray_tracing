#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


#ifdef _DEBUG
	#ifndef __CUDACC__
		#define __CUDACC__
	#endif
	#include "math_functions.h"
#endif

#include <Windows.h>
#include <gl/GL.h>

__host__ __device__
float _maxf(const float a, const float b) {
	float df = a - b;
	int sign = signbit(df);
	df *= sign;
	return a - df;
}
__host__ __device__
float _minf(const float a, const float b) {
	float df = a - b;
	int sign = signbit(df);
	df *= sign;
	return b + df;
}

//******************************************************
//functions dim3 begin
__host__ __device__
float dot(const float3& v1, const float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
__host__ __device__
float dot2(const float3& vec) {
	return dot(vec, vec);
}
__host__ __device__
float3 normalize(const float3& vec) {
	float rl = rsqrt(dot2(vec));
	return float3{ vec.x * rl, vec.y * rl, vec.z * rl };
}
//end
//******************************************************

//******************************************************
//mul dim3 begin
__host__ __device__
float3 operator*(const float3& vec, const float a) {
	return float3{ vec.x * a, vec.y * a, vec.z * a };
}
__host__ __device__
float3 operator/(const float3& vec, const float a) {
	float r = 1 / a;
	return vec * r;
}
__host__ __device__
float3 operator*(const float3& v1, const float3& v2) {
	return float3{
		v1.y * v2.z - v1.z * v2.y,
		v2.x * v1.z - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x
	};
}
//end
//******************************************************

//******************************************************
//mul dim4 begin
__host__ __device__
float4 operator*(const float4& vec, const float a) {
	return float4{ vec.x * a,vec.y * a,vec.z * a,vec.w * a };
}
//end
//******************************************************

//******************************************************
//sum dim2 begin
__host__ __device__
float2 operator*(const float2& vec, const float a) {
	return float2{ vec.x * a, vec.y * a };
}
//end
//******************************************************

//******************************************************
//sum dim3 begin
__host__ __device__
float3 operator+(const float3& v1, const float3& v2) {
	return float3{ v1.x + v2.x,v1.y + v2.y,v1.z + v2.z };
}
__host__ __device__
float3 operator-(const float3& vec) {
	return float3{ -vec.x,-vec.y,-vec.z };
}
__host__ __device__
float3 inline operator-(const float3& v1, const float3& v2) {
	return v1 + -v2;
}
//end
//******************************************************

struct Ray
{
	float3 origin{ 0.f,0.f,0.f };
	float3 direction{ 0.f,0.f,0.f };
};

struct Light
{
	float3 position{ 0.f,0.f,0.f };
	float intensity{ 0 };
};

struct Material
{
	float refractive_index = 0.f;
	float3 diffuse_color{ 0.f,0.f,0.f };
	float4 albedo{ 1.f,0.f,0.f,0.f };
	float spectral_exp = 0.f;
};

struct Sphere
{
	float radius = 0.f;
	float3 color{ 0.f,0.f,0.f };
	float3 pos{ 0.f,0.f,0.f };
	Material material{ 0.f,0.f,0.f };
};

struct intersect_t
{
	float second = 0.f;
	bool first = 0.f;
};

__device__
intersect_t intersect_sphere(const Ray& ray, const Sphere& sphere)
{
	float3 ray_to_center = sphere.pos - ray.origin;

	float b = dot(ray_to_center, ray.direction);
	float c = dot2(ray_to_center) - sphere.radius * sphere.radius;
	float disc = b * b - c;

	float x1 = b - sqrtf(disc), x2 = b + sqrtf(disc);
	float d = x2 - x1;

	int state = 5;
	int mat6x2[6][2] = { {0,0},{1,1},{2,2},{2,0},{1,3},{4,0} };
	
	int sign = signbit(disc);
	state = mat6x2[state][sign];
	sign = signbit(x1);
	state = mat6x2[state][sign];
	sign = signbit(x2);
	state = mat6x2[state][sign];

	intersect_t ress[3] = { {1e20,false},{x1,true},{x2,true} };
	return ress[state];
}

__device__
float3 reflect(const float3& I, const float3& N) {
	float3 result = I - N * 2.f * dot(I, N);
	return result;
}

__device__
float3 refraction(const float3& I, const float3& N, const float eta_t, const float eta_i = 1.f) {
	float cosi = -_maxf(-1.f, _minf(1.f, dot(I, N)));
	if (cosi < 0.f)
		return refraction(I, -N, eta_i, eta_t);
	float eta = eta_i / eta_t;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0.f ? float3{ 1.f,0.f,0.f } : I * eta + N * (eta * cosi - sqrtf(k));
}

__device__
float4 make_color(const Ray& ray, const float3& norm, const float3& obj_c, const float3& lc)
{
	float ambient_strength = 0.2f;
	float3 ambient = lc * ambient_strength;



	float3 color = obj_c * ambient;

	return float4{ color.z,color.y,color.z,1.f };
}

__global__
void kernel_render(float4* result, size_t width, size_t height, const float3 ro)
{
	int x = fmaf(blockIdx.x, blockDim.x, threadIdx.x);
	int y = fmaf(blockIdx.y, blockDim.y, threadIdx.y);
	if (x >= width || y >= height) return;

	float2 uv;
	uv.x = (float)x / width;
	uv.y = (float)y / height;

	float ratio = (float)width / height;
	uv = uv * 2.f;
	uv.x -= 1.f;
	uv.y -= 1.f;
	uv.x *= ratio;

	float3 pixel_pos{ uv.x, uv.y, 0.f };

	Ray ray;
	ray.origin = float3{ 0.f, 0.f, 40.f };
	ray.direction = normalize(pixel_pos - ray.origin);

	const float
		_far = 1e20,
		_near = 1e-4;
	float t = _far;

	Sphere s;
	s.radius = .4f;
	s.pos = float3{ 0.f,0.f,3.f };
	s.color = float3{ .9f,.3f,0.f };

	intersect_t res = intersect_sphere(ray, s);

	int result_index = fmaf(y, width, x);
	if (!res.first) {
		result[result_index] = float4{ uv.x * .35f, uv.y * .35f,.3f,1.f };
		return;
	}

	ray.origin = ray.origin + ray.direction * res.second;
	result[result_index] = make_color(ray, float3{ 1.f,1.f,1.f }, s.color, float3{ 1.f,1.f,1.f });
}

int main()
{
	constexpr size_t
		width = 1920,
		height = 1080;

	HWND main_window = CreateWindowExA(0, "static", "", WS_POPUP | WS_VISIBLE, 0, 0, width, height, 0, 0, 0, 0);
	HDC hDC = GetDC(main_window);

	PIXELFORMATDESCRIPTOR pfd = { sizeof(pfd),1 };
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_SUPPORT_COMPOSITION;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cDepthBits = 24;
	pfd.cAlphaBits = 0;
	pfd.iLayerType = PFD_MAIN_PLANE;


	ShowCursor(false);
	int format = ChoosePixelFormat(hDC, &pfd);
	SetPixelFormat(hDC, format, &pfd);
	HGLRC hGLRC = wglCreateContext(hDC);
	wglMakeCurrent(hDC, hGLRC);

	

	dim3 threads(20, 20);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	float4* device_pixels_color, * host_pixels_color = new float4[height * width];
	float2* host_pixels_pos = new float2[height * width];
	cudaMalloc((void**)&device_pixels_color, height * width * sizeof(float4));

	for (auto i = 0; i < height; i++)
		for (auto j = 0; j < width; j++)
			host_pixels_pos[i * width + j] = float2{ 2.f / width * j - 1.f, 2.f / height * i - 1.f };
		

	for (;;) {
		float3 ray_origin = float3{ 0.f, 0.f, 0.f };

		kernel_render <<< blocks, threads >>> (device_pixels_color, width, height, ray_origin);
		cudaMemcpy(host_pixels_color, device_pixels_color, height * width * sizeof(float4), cudaMemcpyDeviceToHost);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(2, GL_FLOAT, 0, (void*)host_pixels_pos);
		glColorPointer(4, GL_FLOAT, 0, (void*)host_pixels_color);

		glDrawArrays(GL_POINTS, 0, height * width);

		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);

		SwapBuffers(hDC);
		if (GetAsyncKeyState(VK_ESCAPE))
			break;
	}

	ShowCursor(true);
	return 0;
}