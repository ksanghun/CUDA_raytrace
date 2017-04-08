// Visible Spheres - after Sanders and Kandrot CUDA by Example
// raytrace.cu

#include <chrono>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdlib>
//#include <cuda_runtime.h>
//// to remove intellisense highlighting
//#include <device_launch_parameters.h>
//#ifndef __CUDACC__
//#define __CUDACC__
//#endif
//#include <device_functions.h>

//#define DIM 64
//#define DIMDIM (DIM * DIM)
#define IMG_RES 512
#define NTPB 16
#define M_SPHERES 6
#define RADIUS DIM / 10.0f
#define MIN_RADIUS 2.0f
#define rnd(x) ((float) (x) * rand() / RAND_MAX)
#define INF 2e10f
#define M_PI 3.141592653589793
#define INFINITY 1e8
#define MAX_RAY_DEPTH 5


template<typename T>
class Vec3
{
public:
	T x, y, z;

	 Vec3(){}
	void init(){
		x = 0;
		y = 0;
		z = 0;
	}
	 void init(T _v){
		x = _v;
		y = _v;
		z = _v;
	}
	void init(T _x, T _y, T _z){
		x = _x;
		y = _y;
		z = _z;
	}

	Vec3& normalize(){
		T nor2 = length2();
		if (nor2 > 0) {
			T invNor = 1 / sqrt(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}

	Vec3<T> operator * (const T &f) const {
		Vec3<T> t;
		t.init(x * f, y * f, z * f);
		return t;
	}
	 Vec3<T> operator * (const Vec3<T> &v) const {
		Vec3<T> t;
		t.init(x * v.x, y * v.y, z * v.z);
		return t;
	}

	T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
	Vec3<T> operator - (const Vec3<T> &v) const {
		Vec3<T> t;
		t.init(x - v.x, y - v.y, z - v.z);
		return t;
	}
	Vec3<T> operator + (const Vec3<T> &v) const {
		Vec3<T> t;
		t.init(x + v.x, y + v.y, z + v.z);
		return t;
	}


	Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
	Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }

	T length2() const { return x * x + y * y + z * z; }
	T length() const { return sqrt(length2()); }
};

typedef Vec3<float> Vec3f;


class Sphere {
	Vec3f center;                           /// position of the sphere
	float radius, radius2;                  /// sphere radius and radius^2
	Vec3f surfaceColor, emissionColor;      /// surface color and emission (light)
	float transparency, reflection;         /// surface transparency and reflectivity

public:
	Sphere() {}
	void init(Vec3f c, const float r, Vec3f sc, float refl, float transp, Vec3f ec){
		center = c;
		radius = r;
		radius2 = r*r;
		reflection = refl;
		transparency = transp;
		emissionColor = ec;
		surfaceColor = sc;
	}

	Vec3f getCenter() { return center; }
	Vec3f getEmissionCr() { return emissionColor; }
	Vec3f getSurfaceCr() { return surfaceColor; }
	float getTransparency() { return transparency; }
	float getReflection() { return reflection; }

	bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
	{
		Vec3f l = center - rayorig;
		float tca = l.dot(raydir);
		if (tca < 0) return false;
		float d2 = l.dot(l) - tca * tca;

		if (d2 > radius2) return false;
		float thc = sqrt(radius2 - d2);
		t0 = tca - thc;
		t1 = tca + thc;
		return true;
	}


};

float mix(const float &a, const float &b, const float &mix)
{
	return b * mix + a * (1 - mix);
}

void trace(Vec3f &rayorig, Vec3f &raydir, const int &depth, Vec3f* pixel, Sphere* sphere, int k)
{
	float tnear = INFINITY;
	int idx = -1;

	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < M_SPHERES; ++i) {
		float t0 = INFINITY, t1 = INFINITY;
		if (sphere[i].intersect(rayorig, raydir, t0, t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 < tnear) {   // find the closest intersection of speres
				tnear = t0;
				idx = i;
			}
		}
	}

	// if there's no intersection return black or background color
	if (idx<0){
		pixel[k].init(1.0f, 0.5f, 0.5f);
	}
	else{
		Vec3f surfaceColor;
		surfaceColor.init(0);
		Vec3f phit = rayorig + raydir * tnear; // point of intersection
		Vec3f nhit = phit - sphere[idx].getCenter(); // normal at the intersection point
		nhit.normalize(); // normalize normal direction

		float bias = 1e-4; // add some bias to the point from which we will be tracing
		bool inside = false;
		if (raydir.dot(nhit) > 0){
			nhit.x = -nhit.x;
			nhit.y = -nhit.y;
			nhit.z = -nhit.z;
			inside = true;
		}
		if ((sphere[idx].getTransparency() > 0 || sphere[idx].getReflection() > 0) && depth < MAX_RAY_DEPTH) {
			float facingratio = -raydir.dot(nhit);
			
			// change the mix value to tweak the effect
			float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
			
			// compute reflection direction (not need to normalize because all vectors
			// are already normalized)
			Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
			refldir.normalize();
			trace(phit + nhit * bias, refldir, depth + 1, pixel, sphere, k);
			
			Vec3f refraction; 
			refraction.init(0);
			
			// if the sphere is also transparent compute refraction ray (transmission)
			if (sphere[idx].getTransparency()) {
				float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
				float cosi = -nhit.dot(raydir);
				float k = 1 - eta * eta * (1 - cosi * cosi);
				Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
				refrdir.normalize();
				trace(phit + nhit * bias, refldir, depth + 1, pixel, sphere, k);
			}
			// the result is a mix of reflection and refraction (if the sphere is transparent)
			Vec3f reflection = pixel[k];
			surfaceColor = (
				reflection * fresneleffect +
				refraction * (1 - fresneleffect) * sphere[idx].getTransparency()) * sphere[idx].getSurfaceCr();
		}
		else {
			// it's a diffuse object, no need to raytrace any further
			for (unsigned i = 0; i < M_SPHERES; ++i) {
				if (sphere[i].getEmissionCr().x > 0) {
					// this is a light
					Vec3f transmission;
					transmission.init(1);
					Vec3f lightDirection = sphere[i].getCenter() - phit;
					lightDirection.normalize();
					for (unsigned j = 0; j < M_SPHERES; ++j) {
						if (i != j) {
							float t0, t1;
							if (sphere[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
								transmission.init(0.7f);
								break;
							}
						}
					}

					float fCoff = nhit.dot(lightDirection);
					if (fCoff < 0)	fCoff = 0.0f;
					surfaceColor += sphere[idx].getSurfaceCr() * transmission *	fCoff * sphere[i].getEmissionCr();
				}
			}			
		}
		pixel[k] = surfaceColor + sphere[idx].getEmissionCr();
	}

}

void render(float fov, float viewangle, float aspectratio, float iwidth, float iheight, Vec3f* pixel, Sphere* sphere)
{
	for (unsigned y = 0; y < IMG_RES; ++y) {
		for (unsigned x = 0; x < IMG_RES; ++x) {

			int k = x + y * IMG_RES;

			// shared ? //
			float xx = (2 * ((x + 0.5) * iwidth) - 1) * viewangle * aspectratio;
			float yy = (1 - 2 * ((y + 0.5) * iheight)) * viewangle;
			Vec3f raydir, rayorig;
			raydir.init(xx, yy, -1);
			raydir.normalize();
			rayorig.init(0);
			//===========================================//

			// trace //
			trace(rayorig, raydir, 0, pixel, sphere, k);
		}
	}
}


bool SaveImage(char* szPathName, unsigned char* img, int w, int h) {
	// Create a new file for writing
	FILE *f;

	int filesize = 54 + 3 * w*h;  //w is your image width, h is image height, both int

	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	unsigned char bmppad[3] = { 0, 0, 0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	f = fopen(szPathName, "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i<h; i++)
	{
		fwrite(img + (w*(h - i - 1) * 3), 3, w, f);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}
	fclose(f);
	return true;
}

void reportTime(const char* msg, std::chrono::steady_clock::duration span) {
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}

int main(int argc, char* argv[]) {

	Sphere* sphere = new Sphere[M_SPHERES];

	Vec3f center, sc, ec;
	center.init(0.0f, -10004.0f, -20.0f);	sc.init(0.30f, 0.30f, 0.30f);	ec.init(0.0f);
	sphere[0].init(center, 10000.0f, sc, 0.0f, 0.0f, ec);

	center.init(0.0, 0, -20);	sc.init(1.00, 0.32, 0.36);	ec.init(0.1f);
	sphere[1].init(center, 4.0f, sc, 1.0f, 0.5f, ec);

	center.init(5.0, -1, -15);	sc.init(0.10, 0.96, 0.96);	ec.init(0.2f);
	sphere[2].init(center, 2.0f, sc, 1.0f, 0.0f, ec);

	center.init(5.0, 0, -25);	sc.init(0.3f, 0.07, 0.97);	ec.init(0.2f);
	sphere[3].init(center, 3.0f, sc, 1.0f, 0.0f, ec);

	center.init(-5.5, 0, -15);	sc.init(0.10, 0.99, 0.20);	ec.init(0.3f);
	sphere[4].init(center, 3.0f, sc, 1.0f, 0.0f, ec);

	// light
	center.init(-10.0, 50, 30);	sc.init(0.00, 0.00, 0.00);	ec.init(4.0f);
	sphere[5].init(center, 3.0f, sc, 0.0f, 0.0f, ec);
		
	unsigned width = 512, height = 512;
	float invWidth = 1 / float(width), invHeight = 1 / float(height);
	float fov = 45, aspectratio = width / float(height);
	float angle = tan(M_PI * 0.5 * fov / 180.);
	
	Vec3f* a = new Vec3f[IMG_RES*IMG_RES];
	unsigned char* imgbuff = new unsigned char[width*height * 3];
	
	std::chrono::steady_clock::time_point ts, te;
	ts = std::chrono::steady_clock::now();

	for (int i = 0; i < 1000; i++){
		fov += 0.1;
		angle = tan(M_PI * 0.5 * fov / 180.);
		render(fov, angle, aspectratio, invWidth, invHeight, a, sphere);
				
		// save image
		for (unsigned i = 0; i < width * height; ++i) {
			imgbuff[i * 3] = (unsigned char)(std::min(float(1), a[i].x) * 255);
			imgbuff[i * 3 + 1] = (unsigned char)(std::min(float(1), a[i].y) * 255);
			imgbuff[i * 3 + 2] = (unsigned char)(std::min(float(1), a[i].z) * 255);

		}
		char fname[64] = { 0, };
		sprintf(fname, "%d.bmp", i + 1);
		SaveImage(fname, imgbuff, width, height);
	}

	te = std::chrono::steady_clock::now();
	reportTime("Render Time: ", te - ts);
	
	// clean up
	delete[] imgbuff;
	delete[] a;
	delete[] sphere;
}

