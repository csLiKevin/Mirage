
#define ALPHA_THRESHOLD 0.1

#define PI 3.141592654

#define TWOPI 6.283185307

#define INV_PI 0.3183098861

#define INV_TWOPI 0.1591549431

#define SQRT_OF_TWO_OVER_PI 0.7978845608028654

#define LIGHT_CUTOFF 0.001

#if (defined(USE_CLEARCOAT_VALUE) || defined(USE_CLEARCOAT_TEXTURE))

#define USE_CLEARCOAT

#endif
uniform float u_rightEye;uniform vec2 u_size;
#if defined(USE_COMPLEX_SRGB_CONVERSION)
vec3 a(vec3 b){vec3 c=b/12.92;vec3 d=pow((b+0.055)/1.055,2.4);vec3 e=(b<=vec3(0.04045))?c:d;return e;}vec4 f(vec4 b){vec4 c=b/12.92;vec4 d=pow((b+0.055)/1.055,2.4);vec4 e=(b<=vec4(0.04045))?c:d;return e;}vec3 g(vec3 h){vec3 i=h*12.92;vec3 j=(pow(abs(h),1.0/2.4)*1.055)-0.055;vec3 e=(h<=vec3(0.0031308))?i:j;return e;}
#else
vec4 f(vec4 k){return pow(k,vec4(2.2));}vec3 a(vec3 k){return pow(k,vec3(2.2));}vec3 g(vec3 k){return pow(k,vec3(1.0/2.2));}
#endif
float l(vec3 m){return max(dot(m,vec3(0.299,0.587,0.114)),0.0001);}mat3 n(mat3 o){return mat3(o[0][0],o[1][0],o[2][0],o[0][1],o[1][1],o[2][1],o[0][2],o[1][2],o[2][2]);}
#if defined(USE_AMBIENT_IBL)
uniform float u_radianceMipLevels;
#if defined(USE_AMBIENT_DIFFUSE_PROBE_TEXTURE)

#if defined(USE_CUBEMAP_FORMAT)
uniform samplerCube u_irradianceProbeTexture;
#else
uniform sampler2D u_irradianceProbeTexture;
#endif

#elif defined(USE_AMBIENT_DIFFUSE_SH)
uniform vec3 u_irradianceShCoefficients[9];
#endif

#if defined(USE_CUBEMAP_FORMAT)
uniform samplerCube u_radianceTexture;
#else
uniform sampler2D u_radianceTexture;
#endif

#if defined(USE_SPECULAR_POWER_IBL_LOOKUP)
uniform vec3 u_iblConvData;
#endif

#if defined(USE_AMBIENT_SSR)
uniform sampler2D u_ssrTraceTexture;
#endif

#if (defined(USE_PARALLAX_CORRECTION) || defined(USE_ENV_ROTATION))
uniform mat4 u_localizedEnvironmentInverseMatrix;
#endif

#if defined(USE_PARALLAX_CORRECTION)
uniform vec3 u_localizedEnvironmentOffset;uniform vec3 u_localizedEnvironmentMin;uniform vec3 u_localizedEnvironmentMax;
#endif

#if defined(USE_ENV_BRDF_LUT)
uniform sampler2D u_brdfTexture;
#else
vec3 p(vec3 q,float r,float s,float t){vec4 u=vec4(-1.0,-0.0275,-0.572,0.022);vec4 v=vec4(1.0,0.0425,1.04,-0.04);vec4 w=r*u+v;float x=min(w.x*w.x,exp2(-9.28*s))*w.x+w.y;vec2 y=vec2(-1.04,1.04)*x+w.zw;
#if defined(USE_CLEARCOAT)
return q*y.x+y.y*(1.0-t);
#else
return q*y.x+y.y;
#endif
}
#endif

#if defined(USE_BENT_NORMAL_MAP)
uniform sampler2D u_bentNormalTexture;
#endif

#if defined(USE_ENVIRONMENT_REFRACTION)
uniform samplerCube u_environmentRefractionNormalTexture;uniform samplerCube u_environmentRefractionDepthTexture;uniform mat4 u_environmentRefractionMatrix;uniform vec2 u_environmentRefractionNearFar;uniform mat4 u_inverseWorldMatrix;uniform mat4 u_worldMatrix;
#endif

#endif
void z(vec2 A,float B,out vec2 C,out vec2 D,out float E){C=A;C.y*=0.5;D=C;float F=ceil(B);E=fract(B);float G=B-E;float H=exp2(F+1.0);float I=H*0.5;float J=(1.0/I*(H-1.0)-1.0);float K=exp2(G+1.0);float L=K*0.5;float M=(1.0/L*(K-1.0)-1.0);float N=2.0/1024.0;float O=1.0/exp2(F);D*=O;D.y*=1.0-N/O;D.y+=J+N*0.25;float P=1.0/exp2(G);C*=P;C.y*=1.0-N/P;C.y+=M+N*0.25;}vec2 Q(vec3 w){float R=length(w);
#if defined(FLIPPED_LATLONG_Y)
float S=acos(-w.y/R);
#else
float S=acos(w.y/R);
#endif
float T=atan(vec2(w.x),vec2(-w.z)).x;return vec2((PI+T)*INV_TWOPI,S*INV_PI);}vec2 U(vec3 w){float V=2.4;vec2 A;w.y=-w.y;if(w.z<0.0){A=w.xy/(V*(1.0-w.z))+0.5;}else{A=w.xy/(V*(1.0+w.z))+0.5;A.x+=1.0;}A.x*=0.5;return A;}
#if defined(USE_LATLONG_FORMAT)

#if defined(USE_FOVEATED_ENV)
uniform vec2 u_foveaRotation;
#endif
vec4 W(sampler2D X,vec3 Y,float B){
#if defined(USE_ENV_ROTATION)
Y=mat3(u_localizedEnvironmentInverseMatrix)*Y;
#endif
vec2 A=Q(Y);
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
vec2 C;vec2 D;float E;z(A,B,C,D,E);vec4 Z=texture2D(X,C);vec4 ba=texture2D(X,D);return mix(Z,ba,E);}vec2 bb(float bc,float bd,vec2 be,vec2 A){A=fract(A+be);if(A.x>0.5){A.x=(pow(-2.*A.x+2.,1./bc)-2.)/-2.;}else{A.x=pow((2.*A.x)/pow(2.,bc),1./bc);}if(A.y>0.5){A.y=(pow(-2.*A.y+2.,1./bd)-2.)/-2.;}else{A.y=pow((2.*A.y)/pow(2.,bd),1./bd);}return A;}vec4 bf(sampler2D X,vec3 Y){
#if defined(ENV_HAS_MIPMAPS)
return W(X,Y,0.0);
#else
vec2 A=vec2(0.0);float bg=1.0;
#if defined(USE_MONO_180)
A=Q(vec3(Y.z,Y.y,Y.x*-1.0000001));bg=1.0-step(0.5,A.x);A.x*=2.0;
#else
A=Q(Y);
#endif
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
#if defined(USE_FOVEATED_ENV)
A=bb(FOVEA_DISTORTION_EXPONENT_X,FOVEA_DISTORTION_EXPONENT_Y,vec2(INV_TWOPI*u_foveaRotation.x,INV_PI*u_foveaRotation.y),A);
#endif
return texture2D(X,A)*bg;
#endif
}
#elif defined(USE_PARABOLOID_FORMAT)
vec4 bf(sampler2D X,vec3 Y){
#if defined(ENV_HAS_MIPMAPS)
return W(X,Y,0.0);
#else
vec2 A=U(Y);
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
return texture2D(X,A);
#endif
}vec4 W(sampler2D X,vec3 Y,float B){vec2 A=U(Y);
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
vec2 C;vec2 D;float E;z(A,B,C,D,E);vec4 Z=texture2D(X,C);vec4 ba=texture2D(X,D);return mix(Z,ba,E);}
#elif defined(USE_CUBEMAP_FORMAT)
vec3 bh(vec3 k,float bi,float B){float bj=max(max(abs(k.x),abs(k.y)),abs(k.z));float bk=1.0-exp2(B)/bi;if(abs(k.x)!=bj) k.x*=bk;if(abs(k.y)!=bj) k.y*=bk;if(abs(k.z)!=bj) k.z*=bk;return k;}
#if defined(ENV_HAS_MIPMAPS)

#extension GL_EXT_shader_texture_lod : enable

#endif
vec4 bf(samplerCube X,vec3 Y){
#if defined(ENV_HAS_MIPMAPS)
return textureCubeLodEXT(X,Y,0.0);
#else
return textureCube(X,Y);
#endif
}vec4 W(samplerCube X,vec3 Y,float B){
#if defined(USE_ENV_ROTATION)
Y=mat3(u_localizedEnvironmentInverseMatrix)*Y;
#endif
#if defined(ENV_HAS_MIPMAPS)
return textureCubeLodEXT(X,Y,B);
#else
return textureCube(X,Y);
#endif
}
#endif
float bl(float bm){return floor(log2(bm));}float bn(float r){return (2.0/(r*r))-1.0;}float bo(float bp,float bq,float br,float bs){return (bq-1.0-br)-log2(bp)*bs;}float bt(float bp,float bq,float bu){float bv=(exp2(-10.0/sqrt(bp))-0.00098)/0.9921;return (bq-1.0)*(1.0-clamp(bv/bu,0.0,1.0));}vec2 bw(float bx){bx*=256.0;float x=floor(bx);bx=(bx-x)*256.0;float y=floor(bx);x*=0.00390625;y*=0.00390625;return vec2(x,y);}float by(vec2 k){return k.x+(k.y*0.00390625);}vec4 bz(float bx){vec4 bA=vec4(1.0,255.0,65025.0,160581375.0)*bx;bA=fract(bA);bA-=bA.yzww*vec4(1.0/255.0,1.0/255.0,1.0/255.0,0.0);return bA;}float bB(vec4 bC){return dot(bC,vec4(1.0,1.0/255.0,1.0/65025.0,1.0/160581375.0));}float bD(vec3 bE){return dot(bE,vec3(1.0,1.0/255.0,1.0/65025.0));}vec2 bF(vec3 bG){vec2 e=normalize(bG.xy)*(sqrt(bG.z*0.5+0.5));return e;}vec3 bH(vec2 k){vec3 bG;bG.z=dot(k,k)*2.0-1.0;bG.xy=normalize(k)*sqrt(1.0-bG.z*bG.z);return bG;}float bI(vec3 bG,vec4 bJ){float bK=0.0;float bL=sqrt(1.5/PI);bK+=bJ.x*(1.0/sqrt(2.0*PI));bK+=bJ.y*-bL*bG.y;bK+=bJ.z*bL*(2.0*bG.z-1.0);bK+=bJ.w*-bL*bG.x;return bK;}
#if !defined(USE_VERTEX_TANGENT_SPACE) && (defined(USE_NORMAL_MAP) || defined(USE_STATIC_AMBIENT_DIRECTIONAL_OCCLUSION) || defined(USE_ANISOTROPIC_BRDF) || defined(USE_BENT_NORMAL_MAP))
mat3 bM(vec3 bN,vec3 bO,vec2 bP){
#extension GL_OES_standard_derivatives : enable
vec3 bQ=dFdx(bO);vec3 bR=dFdy(bO);vec2 bS=dFdx(bP);vec2 bT=dFdy(bP);vec3 bU=cross(bR,bN);vec3 bV=cross(bN,bQ);vec3 bW=bU*bS.x+bV*bT.x;vec3 bX=bU*bS.y+bV*bT.y;float bY=inversesqrt(max(dot(bW,bW),dot(bX,bX)));return mat3(bW*bY,bX*bY,bN);}
#endif

#if defined(USE_LOGLUV_INPUT)
const mat3 bj=mat3(0.2209,0.3390,0.4184,0.1138,0.6780,0.7319,0.0102,0.1130,0.2969);vec4 bZ(vec3 ca){vec4 cb;vec3 cc=bj*ca;cc=max(cc,vec3(1e-6,1e-6,1e-6));cb.xy=cc.xy/cc.z;float cd=2.0*log2(cc.y)+127.0;cb.w=fract(cd);cb.z=(cd-(floor(cb.w*255.0))/255.0)/255.0;return cb;}const mat3 ce=mat3(6.0013,-2.700,-1.7995,-1.332,3.1029,-5.7720,.3007,-1.088,5.6268);vec3 cf(vec4 cg){float cd=cg.z*255.0+cg.w;vec3 cc;cc.y=exp2((cd-127.0)/2.0);cc.z=cc.y/cg.y;cc.x=cg.x*cc.z;vec3 ca=ce*cc;ca.rg=ca.gr;return max(ca,0.0);}
#endif
vec4 ch(vec3 m){vec4 ci;m*=1.0/6.0;ci.a=clamp(max(max(m.r,m.g),max(m.b,1e-6)),0.0,1.0);ci.a=ceil(ci.a*255.0)/255.0;ci.rgb=m/ci.a;return ci;}vec3 cj(vec4 ci,float ck){return ck*ci.a*ci.rgb;}vec3 cl(vec4 ci,float ck,out float cm){cm=ceil(ci.a-0.5);float cn=(ci.a-0.5*cm);return ck*cn*ci.rgb;}
#if defined(USE_RGBM_INPUT)
uniform float u_rgbmMaxRange;
#endif

#if defined(USE_RGBM_IRRADIANCE_PROBE_INPUT)
uniform float u_rgbmIrradianceProbeMaxRange;
#endif

#if defined(USE_RGBM_RADIANCE_PROBE_INPUT)
uniform float u_rgbmRadianceProbeMaxRange;
#endif

#if defined(USE_RGBM_IRRADIANCE_INPUT)
uniform float u_rgbmIrradianceMaxRange;
#endif

#if defined(USE_RGBM_DIFFUSE_INPUT)
uniform float u_rgbmDiffuseMaxRange;
#endif

#if defined(USE_RGBM_NORMAL_INPUT)
uniform float u_rgbmNormalMaxRange;
#endif

#if !defined(USE_HDR)

#if defined(USE_COLOR_GRADIENT)
uniform vec4 u_gradientColorA;uniform vec4 u_gradientColorB;uniform vec2 u_gradientDirection;vec3 co(vec3 m){vec4 cp=mix(u_gradientColorA,u_gradientColorB,dot(gl_FragCoord.xy/u_size,u_gradientDirection));return (1.0-cp.a)*m+cp.a*cp.rgb;}
#endif
vec3 cq(vec3 m){m=max(vec3(0),m-vec3(0.004));m=(m*(6.2*m+vec3(0.5)))/(m*(6.2*m+vec3(1.7))+vec3(0.06));m=min(vec3(1.0,1.0,1.0),m);return m;}vec3 cr(vec3 m){return pow(m,vec3(1.0/2.2));}vec3 cs(vec3 m){
#if defined(USE_TONEMAP_FILMIC)
m=cq(m);
#else
m=cr(m);
#endif
return m;}
#define PI_FOURTH 0.78539816339
uniform float u_fStop;uniform float u_shutterSpeed;uniform float u_lensTransmittance;uniform float u_filmSpeed;
#if defined(USE_VIGNETTING)
uniform float u_focalLength;uniform float u_opticalVignettingStrength;uniform float u_distanceToFilm;uniform vec2 u_filmSize;
#endif
float ct(vec2 cu){float bK=1.0;
#if defined(USE_VIGNETTING)
float cv=length(u_filmSize*cu);float cw=cos(atan(cv/u_distanceToFilm));bK=cw*cw*cw*cw;
#endif
return bK;}float cx(float cy){float bK=1.0;
#if defined(USE_VIGNETTING)
float cz=u_fStop*u_focalLength/(100.0*u_opticalVignettingStrength);float cA=smoothstep(0.0,cz,cy);bK=1.0-cA;
#endif
return bK;}float cB(vec2 cu){float cC=ct(cu);return PI_FOURTH*u_filmSpeed*u_lensTransmittance*cC*u_shutterSpeed/(u_fStop*u_fStop);}
#endif
vec3 cD(const in vec3 cE){const mat3 cF=mat3(0.25,0.5,-0.25,0.5,0.0,0.5,0.25,-0.5,-0.25);return cF*cE;}vec3 cG(const in vec3 cH){vec3 cE;cE.r=cH[0]+cH[1]-cH[2];cE.g=cH[0]+cH[2];cE.b=cH[0]-cH[1]-cH[2];return cE;}vec3 cI(vec3 cJ,vec3 cK,vec3 cL){return cJ-dot(cJ-cK,cL)*cL;}int cM(in vec3 bO,in vec3 cN,in vec3 cO){if(dot(bO-cN,cO)>=0.0) return 1;else return 0;}vec3 cP(in vec3 cQ,in vec3 cR,in vec3 cN,in vec3 cO){return cQ+cR*(dot(cO,cN-cQ)/dot(cO,cR));}bool cS(vec3 bG,vec3 cT,vec3 cU,vec3 R,out float cV,out float cW){float cX=dot(bG,R);if(cX>0.000001){vec3 cY=cT-cU;cV=dot(cY,bG)/cX;vec3 bO=cU+R*cV;vec3 k=bO-cT;cW=dot(k,k);return cV>=0.0;}return false;}vec2 cZ(vec3 da,vec3 Y,vec3 db,vec3 dc){vec3 dd=1.0/(Y);vec3 de=(db-da)*dd;vec3 df=(dc-da)*dd;vec3 dg=min(de,df);vec3 dh=max(de,df);return vec2(max(dg.x,max(dg.y,dg.z)),min(dh.x,min(dh.y,dh.z)));}vec3 di(vec3 k){return k*k;}vec3 dj(vec3 q,vec3 k,vec3 dk){float dl=clamp(dot(k,dk),0.0,1.0);vec3 dm=vec3(dl);vec3 dn=vec3(1.0);vec3 dp=sqrt(clamp(vec3(0.0),vec3(0.99,0.99,0.99),q));vec3 bG=(dn+dp)/(dn-dp);vec3 cp=sqrt(bG*bG+vec3(dl*dl-1.0));return 0.5*di((cp-dm)/(cp+dm))*(dn+di(((cp+dm)*dl-dn)/((cp-dm)*dl+dn)));}float dq(float dr,float dl){float ds=pow(1.0-dl,5.0);return ds+(1.0-ds)*dr;}vec3 dt(vec3 dr,float du,float dl){return dr+(vec3(du)-dr)*pow(1.0-dl,5.0);}float dv(float dr,float du,float dl){return dr+(du-dr)*pow(1.0-dl,5.0);}float dw(vec3 dr){return clamp(50.0*dot(dr,vec3(0.33)),0.0,1.0);}float dx(float dr){return clamp(50.0*dr,0.0,1.0);}vec3 dy(vec3 dz[9],vec3 Y){float dA=Y.x*Y.x,dB=Y.y*Y.y,dC=Y.z*Y.z;float dD=Y.x*Y.y,dE=Y.y*Y.z,dF=Y.x*Y.z;vec3 dG=dz[0];vec3 dH=dz[1];vec3 dI=dz[2];vec3 dJ=dz[3];vec3 dK=dz[4];vec3 dL=dz[5];vec3 dM=dz[6];vec3 dN=dz[7];vec3 dO=dz[8];vec3 bK=0.429043*dO*(dA-dB)+0.743125*dM*dC+0.886227*dG-0.247708*dM+2.0*0.429043*(dK*dD+dN*dF+dL*dE)+2.0*0.511664*(dJ*Y.x+dH*Y.y+dI*Y.z);return max(bK,vec3(0.0));}
#if defined(PARTICLE)
uniform sampler2D u_baseTexture;varying vec2 v_texCoord0;varying vec4 v_color;void main(){
#if defined(USE_GAMMA_SPACE)
vec4 bK=f(texture2D(u_baseTexture,v_texCoord0));
#else
vec4 bK=texture2D(u_baseTexture,v_texCoord0);
#endif
#if defined(DISCARD_ALPHA)
if(bK.a<ALPHA_THRESHOLD){discard;}
#endif
bK*=v_color;
#if !defined(USE_HDR)
vec2 cu=(gl_FragCoord.xy/u_size)-vec2(0.5,0.5);float cv=dot(cu,cu);
#if defined(USE_COLOR_GRADIENT)
bK.rgb=co(bK.rgb);
#endif
bK.rgb*=cB(cu)*cx(cv);bK.rgb=cs(bK.rgb);
#endif
gl_FragColor=bK;}
#elif (defined(LIT) || defined(LIT_POINT_SPRITE) || defined(LIT_PARTICLE))

#if (defined(USE_CUBE_IBR) || defined(USE_LATLONG_IBR))
varying vec3 v_direction;
#if defined(USE_DEFERRED_TEXTURING)

#define SAMPLER_LOOKUP texture2D

#define SAMPLER_TEXCOORD texCoord0

#define SAMPLER_BAKED_TEXCOORD bakedTc

#define SAMPLER_TYPE sampler2D

#define SAMPLER_COORD_TYPE vec2

#if defined(USE_LATLONG_IBR)
uniform sampler2D u_materialIdTexture;uniform sampler2D u_uvTexture;
#elif defined(USE_CUBE_IBR)
uniform samplerCube u_materialIdTexture;uniform samplerCube u_uvTexture;
#endif

#elif defined(USE_LATLONG_IBR)

#define SAMPLER_LOOKUP texture2D

#define SAMPLER_TEXCOORD worldToLatLong(texCoord0)

#define SAMPLER_BAKED_TEXCOORD worldToLatLong(bakedTc)

#define SAMPLER_TYPE sampler2D

#define SAMPLER_COORD_TYPE vec3

#elif defined(USE_CUBE_IBR)

#define SAMPLER_LOOKUP textureCube

#define SAMPLER_TEXCOORD texCoord0

#define SAMPLER_BAKED_TEXCOORD bakedTc

#define SAMPLER_TYPE samplerCube

#define SAMPLER_COORD_TYPE vec3

#endif

#else

#define SAMPLER_LOOKUP texture2D

#define SAMPLER_TEXCOORD texCoord0

#define SAMPLER_BAKED_TEXCOORD bakedTc

#define SAMPLER_TYPE sampler2D

#define SAMPLER_COORD_TYPE vec2

#endif

#define ENV_CORRECTION_ITERATION_COUNT 3
float dP(samplerCube dQ,vec3 dR,vec2 dS){
#if defined(USE_DEPTH_PACKING)
return dS.x+bB(textureCube(dQ,dR))*(dS.y-dS.x);
#elif defined(USE_DEPTH_24_PACKING)
return dS.x+bD(textureCube(dQ,dR).rgb)*(dS.y-dS.x);
#else
return dS.x+textureCube(dQ,dR).x*(dS.y-dS.x);
#endif
}float dP(sampler2D dQ,vec3 dR,vec2 dS){
#if defined(USE_DEPTH_PACKING)
return dS.x+bB(texture2D(dQ,Q(dR)))*(dS.y-dS.x);
#elif defined(USE_DEPTH_24_PACKING)
return dS.x+bD(texture2D(dQ,Q(dR)).rgb)*(dS.y-dS.x);
#else
return dS.x+texture2D(dQ,Q(dR)).x*(dS.y-dS.x);
#endif
}vec3 dT(vec3 cJ,vec3 dU,sampler2D dV,vec2 dS,inout float dW){float dX=dP(dV,dU,dS);float dY=dX-dot(cJ,dU);vec3 bO=cJ+dU*dY;float dZ=length(bO)/dP(dV,bO,dS);float ea=0.0,eb=0.0,ec=dZ,ed=dZ;if(dZ<1.0) ea=dY;else eb=dY;float ee=max(dY+dX*(1.0-dZ),0.0);vec3 R=cJ+dU*ee;for(int ef=0;ef<ENV_CORRECTION_ITERATION_COUNT;ef++){float eg;float eh=length(R)/dP(dV,R,dS);if(eh<1.0){ea=ee;ec=eh;eg=(eb==0.0)?dX*(1.0-eh):(ee-eb)*(1.0-eh)/(eh-ed);}else{eb=ee;ed=eh;eg=(ea==0.0)?dX*(1.0-eh):(ee-ea)*(1.0-eh)/(eh-ec);}float ei=ee;ee=max(ee+eg,0.0);R=cJ+dU*ee;}return R;}
#if defined(USE_REFRACTION)
uniform mat4 u_projectionMatrix;
#endif
vec3 dT(vec3 cJ,vec3 dU,samplerCube dV,vec2 dS,inout float dW){float dX=dP(dV,dU,dS);float dY=dX-dot(cJ,dU);vec3 bO=cJ+dU*dY;float dZ=length(bO)/dP(dV,bO,dS);float ea=0.0,eb=0.0,ec=dZ,ed=dZ;if(dZ<1.0) ea=dY;else eb=dY;float ee=max(dY+dX*(1.0-dZ),0.0);vec3 R=cJ+dU*ee;for(int ef=0;ef<ENV_CORRECTION_ITERATION_COUNT;ef++){float eg;float eh=length(R)/dP(dV,R,dS);if(eh<1.0){ea=ee;ec=eh;eg=(eb==0.0)?dX*(1.0-eh):(ee-eb)*(1.0-eh)/(eh-ed);}else{eb=ee;ed=eh;eg=(ea==0.0)?dX*(1.0-eh):(ee-ea)*(1.0-eh)/(eh-ec);}float ei=ee;ee=max(ee+eg,0.0);R=cJ+dU*ee;}return R;}
#if (SHADOWED_DIR_LIGHT_COUNT > 0)

#define HAS_SHADOWED_LIGHTS

#endif

#if (defined(USE_METALLIC_TEXTURE) || defined(USE_METALLIC_VALUE) || defined(USE_IRRADIANCE_WITH_METALLIC) || defined(USE_DIFFUSE_TEXTURE_WITH_METALLIC))

#define USE_METALLIC_WORKFLOW

#endif

#if (defined(LIT) || defined(LIT_PARTICLE))
varying vec2 v_texCoord0;
#if defined(USE_2ND_UV_FOR_BAKED_MAPS)
varying vec2 v_texCoord1;
#endif

#if defined(USE_UV_BASED_RADIAL_OPACITY)
varying vec2 v_originalTexCoord0;
#endif

#if defined(USE_DIFFUSE_SPRITE_ANIMATION)
varying vec2 v_texCoord2;uniform sampler2D u_diffuseSpriteAnimationTexture;uniform float u_diffuseSpriteAnimationAlpha;
#endif

#endif

#if defined(LIT_POINT_SPRITE)
varying float v_radius;
#endif
varying vec3 v_position;
#if defined(LIT_POINT_SPRITE)
uniform mat4 u_inverseViewMatrix;
#endif
uniform vec3 u_cameraLocation;
#if (defined(HAS_SHADOWED_LIGHTS) || defined(USE_Z_DEPTH_FOG))
varying vec3 v_positionCS;
#endif

#if (defined(USE_DIFFUSE_TEXTURE) || defined(USE_DIFFUSE_TEXTURE_WITH_ALPHA) || defined(USE_DIFFUSE_TEXTURE_WITH_OCCLUSION) || defined(USE_DIFFUSE_TEXTURE_WITH_METALLIC))
uniform SAMPLER_TYPE u_diffuseTexture;
#elif defined(USE_DIFFUSE_COLOR)
uniform vec3 u_diffuseColor;
#endif

#if (defined(USE_STATIC_AMBIENT_OCCLUSION) && !defined(USE_DIFFUSE_TEXTURE_WITH_OCCLUSION))

#if defined(USE_VERTEX_AO)
varying vec3 v_ao;
#else
uniform SAMPLER_TYPE u_aoTextureStatic;
#endif

#endif

#if defined(USE_DYNAMIC_AMBIENT_OCCLUSION)
uniform sampler2D u_aoTextureDynamic;
#endif

#if defined(USE_VERTEX_NORMAL)
varying vec3 v_normal;
#else

#extension GL_OES_standard_derivatives : enable

#endif

#if defined(USE_NORMAL_MAP)
uniform SAMPLER_TYPE u_normalTexture;uniform float u_normalMapStrength;
#endif

#if (defined(USE_VERTEX_TANGENT_SPACE))
varying vec3 v_tangent;varying vec3 v_bitangent;
#endif

#if defined(USE_SPECULAR_TEXTURE)
uniform SAMPLER_TYPE u_specularTexture;
#elif defined(USE_SPECULAR_COLOR)
uniform vec3 u_specularColor;
#elif defined(USE_METALLIC_TEXTURE)
uniform SAMPLER_TYPE u_metallicTexture;
#elif defined(USE_METALLIC_VALUE)
uniform float u_metallicValue;
#endif

#if defined(USE_ROUGHNESS_TEXTURE)
uniform SAMPLER_TYPE u_roughnessTexture;
#elif defined(USE_ROUGHNESS_VALUE)
uniform float u_roughness;
#endif

#if defined(USE_WETNESS_TEXTURE)
uniform SAMPLER_TYPE u_wetnessTexture;
#elif defined(USE_WETNESS_VALUE)
uniform float u_wetness;
#endif

#if defined(USE_ANISOTROPY_TEXTURE)
uniform SAMPLER_TYPE u_anisotropyTexture;
#elif defined(USE_ANISOTROPY_VALUE)
uniform float u_anisotropy;
#endif

#if defined(USE_EMISSIVE_TEXTURE)
uniform SAMPLER_TYPE u_emissiveTexture;
#if defined(USE_RGBM_EMISSIVE_INPUT)
uniform float u_rgbmEmissiveMaxRange;
#endif

#elif defined(USE_EMISSIVE_COLOR)
uniform vec3 u_emissiveColor;
#endif

#if defined(USE_EMISSIVE_TEXTURE) || defined(USE_EMISSIVE_COLOR)
uniform float u_emissiveStrength;
#endif

#if defined(USE_CLEARCOAT)
uniform float u_clearcoatRoughness;
#endif

#if defined(USE_CLEARCOAT_VALUE)
uniform float u_clearcoatValue;
#elif defined(USE_CLEARCOAT_TEXTURE)
uniform SAMPLER_TYPE u_clearcoatTexture;
#endif

#if defined(USE_HIGHLIGHT_COLOR)
uniform vec4 u_highlightColor;uniform vec3 u_highlightPosT;
#endif

#if (defined(USE_OIT_TRANSPARENCY) || defined(USE_TRANSPARENCY))

#if defined(USE_OPACITY_TEXTURE)
uniform SAMPLER_TYPE u_opacityTexture;
#elif defined(USE_OPACITY_VALUE)
uniform float u_opacity;
#endif

#endif

#if (defined(HAS_SHADOWED_LIGHTS) || defined(USE_REFRACTION))
uniform mat4 u_viewMatrix;
#endif

#if defined(USE_REFRACTION)
uniform sampler2D u_refractionTexture;
#endif

#if defined(USE_AMBIENT_COLOR)
uniform vec3 u_ambientColor;
#endif

#if defined(USE_HDR)

#if defined(USE_HIGH_LUMINANCE_DIFFUSE_BOOST)
uniform vec2 u_highLuminanceDiffuseBoostParams;
#endif

#endif

#if defined(USE_AMBIENT_DIFFUSE_TEXTURE)
uniform SAMPLER_TYPE u_irradianceTexture;uniform float u_irradianceTextureBoost;
#if defined(USE_IRRADIANCE_BLEND)
uniform SAMPLER_TYPE u_irradianceTextureBlendDst;uniform float u_irradianceTextureBlendFactor;
#if defined(USE_RGBM_IRRADIANCE_INPUT)
uniform float u_rgbmIrradianceMaxRangeBlendDst;
#endif

#endif

#endif

#if DIR_LIGHT_COUNT > 0
uniform vec3 u_directionalLightColor[DIR_LIGHT_COUNT];uniform vec3 u_directionalLightDirection[DIR_LIGHT_COUNT];uniform float u_directionalLightRadius[DIR_LIGHT_COUNT];
#endif

#if POINT_LIGHT_COUNT > 0
uniform vec3 u_pointLightPosition[POINT_LIGHT_COUNT];uniform vec3 u_pointLightColor[POINT_LIGHT_COUNT];uniform float u_pointLightFalloff[POINT_LIGHT_COUNT];
#endif

#if SPOT_LIGHT_COUNT > 0
uniform vec3 u_spotLightPosition[SPOT_LIGHT_COUNT];uniform vec3 u_spotLightDirection[SPOT_LIGHT_COUNT];uniform float u_spotLightFalloff[SPOT_LIGHT_COUNT];uniform float u_spotLightAngle[SPOT_LIGHT_COUNT];uniform float u_spotLightExponent[SPOT_LIGHT_COUNT];uniform vec3 u_spotLightColor[SPOT_LIGHT_COUNT];uniform sampler2D u_spotLightGoboTexture[SPOT_LIGHT_COUNT];uniform bool u_spotLightHasGobo[SPOT_LIGHT_COUNT];varying vec4 v_spotLightProjPosition[SPOT_LIGHT_COUNT];
#endif

#if SHADOWED_DIR_LIGHT_COUNT > 0
uniform vec3 u_shadowedDirectionalLightColor[SHADOWED_DIR_LIGHT_COUNT];uniform vec3 u_shadowedDirectionalLightDirection[SHADOWED_DIR_LIGHT_COUNT];uniform float u_shadowedDirectionalLightRadius[SHADOWED_DIR_LIGHT_COUNT];uniform sampler2D u_shadowedDirectionalLightShadowMap[SHADOWED_DIR_LIGHT_COUNT];uniform vec2 u_shadowedDirectionalLightShadowMapSize[SHADOWED_DIR_LIGHT_COUNT];uniform mat4 u_shadowedDirectionalLightShadowMapViewProjectionMatrix[SHADOWED_DIR_LIGHT_COUNT];
#endif

#if SPHERE_LIGHT_COUNT > 0
uniform vec3 u_sphereLightColor[SPHERE_LIGHT_COUNT];uniform vec3 u_sphereLightPos[SPHERE_LIGHT_COUNT];uniform float u_sphereLightFalloff[SPHERE_LIGHT_COUNT];uniform float u_sphereLightRadius[SPHERE_LIGHT_COUNT];
#endif

#if AREA_LIGHT_COUNT > 0
uniform vec3 u_areaLightPosition[AREA_LIGHT_COUNT];uniform vec3 u_areaLightNormal[AREA_LIGHT_COUNT];uniform vec3 u_areaLightUp[AREA_LIGHT_COUNT];uniform vec3 u_areaLightRight[AREA_LIGHT_COUNT];uniform vec2 u_areaLightSize[AREA_LIGHT_COUNT];uniform vec3 u_areaLightColor[AREA_LIGHT_COUNT];
#endif
float ej(float o,float ek){float el=ek*ek;float em=sqrt((1.0-el)/el);float en=1.0/(o*em);float eo=en*en;float cp=1.0;if(en<1.6) cp*=(3.535*en+2.181*eo)/(1.0+2.276*en+2.577*eo);return cp;}float ep(float dl){return 1.0/(4.0*dl*dl);}float eq(float o,float s,float er){float es=o*o;float et=s+sqrt(s*(s-s*es)+es);float eu=er+sqrt(er*(er-er*es)+es);return 1.0/(et*eu);}float ev(float o,float s,float er){float et=er*(s*(1.0-o)+o);float eu=s*(er*(1.0-o)+o);return 0.5/(et+eu);}float ew(float o,float s,float er){float ex=o*0.5;float et=s*(1.0-ex)+ex;float eu=er*(1.0-ex)+ex;return 0.25/(et*eu);}float ey(float o,float s,float er){float ez=ej(o,er);float eA=ej(o,s);return ez*eA;}float eB(float o,float eC){float es=o*o;float cX=(eC*es-eC)*eC+1.0;return es/(PI*cX*cX);}float eD(float o,float eC){float es=o*o;float eE=eC*eC;return exp((eE-1.0)/(es*eE))/(PI*es*eE*eE);}float eF(float eG,float eH,float eC,vec3 eI,vec3 eJ,vec3 eK){float eL=dot(eJ,eI);float eM=dot(eK,eI);float cW=eL*eL/(eG*eG)+eM*eM/(eH*eH)+eC*eC;return 1.0/(PI*eG*eH*cW*cW);}float eN(float o,float eC){float eE=eC*eC;float eO=eE*eE;float es=o*o;float eP=(1.0-eE)/eE;float eQ=exp(-eP/es);return eQ/(PI*es*eO);}vec3 eR(vec3 eS){return eS*INV_PI;}vec3 eT(vec3 eS,float r,float s,float er,float dl){float eU=1.0-s;float eV=eU*eU;eV=eV*eV*eU;float eW=1.0-er;float eX=eW*eW;eX=eX*eX*eW;float eY=(0.5+2.0*dl*dl)*r;float eZ=1.0+(eY-1.0)*eV;float fa=1.0+(eY-1.0)*eX;return eS*INV_PI*eZ*fa*(1.0-0.3333*r);}vec3 fb(vec3 eS,float o,float s,float er,float dl){float fc=2.0*dl-1.0;float es=o*o;float fd=1.0-0.5*es/(es+0.33);float fe=fc-s*er;float ff=0.45*es/(es+0.09)*fe*(fe>=0.0?min(1.0,er/s):er);return eS*INV_PI*(er*fd+ff);}vec3 fg(vec3 k,vec3 R,vec3 fh,vec3 bG,float r,float fi,vec3 q,vec3 eS,vec3 fj,vec3 fk,float s,float du,float t,float fl){vec3 dk=normalize(k+R);vec3 fm=normalize(k+fh);float o=r*r;float er=clamp(dot(bG,R),0.0,1.0);float fn=clamp(dot(bG,fh),0.0,1.0);float fo=clamp(dot(bG,fm),0.0,1.0);float dl=clamp(dot(k,dk),0.0,1.0);float fp=clamp(dot(k,fm),0.0,1.0);
#if defined(USE_SCHLICK_FRESNEL)
vec3 bx=dt(q,du,fp);
#elif defined(USE_REFERENCE_FRESNEL)
vec3 bx=dj(q,fp);
#else
vec3 bx=q;
#endif
float cW=1.0;
#if defined(USE_STANDARD_BRDF)
#if defined(USE_GGX_DISTRIBUTION)
cW=eB(o,fo);
#elif defined(USE_BECKMANN_DISTRIBUTION)
cW=eD(o,fo);
#endif
#elif defined(USE_ANISOTROPIC_BRDF)
cW=eF(o,mix(0.0,o,1.0-fi),fo,bG,fj,fk);
#elif defined(USE_CLOTH_BRDF)
cW=eN(o,fo);
#endif
float fq=0.25;
#if (defined(USE_STANDARD_BRDF) || defined(USE_ANISOTROPIC_BRDF))
#if defined(USE_SCHLICK_VISIBILITY)
fq=ew(o,s,fn);
#elif defined(USE_SMITH_VISIBILITY)
fq=eq(o,s,fn);
#elif defined(USE_SMITHJOINT_VISIBILITY)
fq=ev(o,s,fn);
#endif
#elif defined(USE_CLOTH_BRDF)
fq=ey(o,s,fn);
#endif
vec3 fr=vec3(0.0);
#if defined(USE_LAMBERT_DIFFUSE)
fr=eR(eS);
#elif defined(USE_BURLEY_DIFFUSE)
fr=eT(eS,r,s,er,dl);
#elif defined(USE_ORENNAYAR_DIFFUSE)
fr=fb(eS,r,s,er,dl);
#endif
#if defined(USE_CLEARCOAT)
float fs=eB(fl*fl,fo);float ft=dq(0.04,fp);ft*=t;float fu=(1.0-ft);float fv=ep(fp);return (fr+cW*fq*bx)*fu+fs*fv*ft;
#else
return fr+cW*fq*bx;
#endif
}float fw(float db,float dc,float k){return clamp((k-db)/(dc-db),0.0,1.0);}float fx(float fy,float fz){return fw(fz,1.0,fy);}float fA(sampler2D fB,float fC,float fD,vec2 fE,vec2 fF){
#if defined(USE_HDR)
vec2 fG=texture2D(fB,fE).xy;
#else
vec4 fH=texture2D(fB,fE);vec2 fG=vec2(by(fH.xy),by(fH.zw));
#endif
float bO=float((fC<=fG.x));float fI=fG.y-(fG.x*fG.x);fI=max(fI,0.0001);float cW=fC-fG.x;float fy=fI/(fI+cW*cW);return max(bO,fx(fy,0.5));}float fJ(sampler2D fB,vec2 fF,mat4 fK,vec3 fL,float fD){vec4 fM=vec4(fL,1.0);mat4 fN=fK;vec4 fO=fN*fM;fO.xyz/=fO.w;float fC=fO.z;vec2 fE=fO.xy*0.5+0.5;fE+=(0.5/fF);float fP=fA(fB,fC,fD,fE,fF);return fP;}vec3 fQ(vec3 fR,vec3 fS,float fT,vec3 fU,vec3 fV,float fW,float fX,vec3 fY,vec3 fZ,vec3 ga,float gb,vec3 fj,vec3 fk,float s,float du,float gc,float gd){float er=clamp(dot(fY,fR),0.0,1.0);if(er>0.0){float w=sin(fT);float cW=cos(fT);float ge=dot(fR,ga);vec3 gf=ga-ge*fR;vec3 gg=ge<cW?normalize(cW*fR+normalize(gf)*w):ga;vec3 bx=fg(fZ,fR,gg,fY,fW,fX,fV,fU,fj,fk,s,du,gc,gd);float gh=pow(max(1.0-fT,0.0),2.0);return gh*gb*er*bx*fS;}return vec3(0.0);}vec3 gi(vec3 gj,float gk,vec3 fS,vec3 gl,vec3 fY,vec3 fU,vec3 fV,float fW,float fX,vec3 fZ,vec3 fj,vec3 fk,float s,float du,float gc,float gd){vec3 R=gj-gl;float cv=length(R);R/=cv;float er=clamp(dot(fY,R),0.0,1.0);if(er>0.0){float gm=pow(clamp(1.0-pow(cv/gk,4.0),0.0,1.0),2.0)/(cv*cv+1.0);vec3 bx=fg(fZ,R,R,fY,fW,fX,fV,fU,fj,fk,s,du,gc,gd);return gm*er*bx*fS;}return vec3(0.0);}vec3 gn(vec3 gj,vec3 fR,float go,float gk,float gp,vec3 fS,vec4 gq,sampler2D gr,bool gs,vec3 gl,vec3 fY,vec3 fU,vec3 fV,float fW,float fX,vec3 fZ,vec3 fj,vec3 fk,float s,float du,float gc,float gd){vec3 R=gj-gl;float cv=length(R);R/=cv;float gt=dot(fR,R);if(gt>go){float er=clamp(dot(fY,fR),0.0,1.0);if(er>0.0){float gm=pow(clamp(1.0-pow(cv/gk,4.0),0.0,1.0),2.0)/(cv*cv+1.0);gt=pow(gt,gp);vec3 bx=fg(fZ,R,R,fY,fW,fX,fV,fU,fj,fk,s,du,gc,gd);if(gs){float gu=smoothstep(0.4,0.6,texture2D(gr,(gq.xy/gq.w)*0.5+0.5).x);return gu*gt*er*bx*fS;}else{return gm*gt*er*bx*fS;}}}return vec3(0.0);}vec3 gv(vec3 gj,float fT,float gk,vec3 fS,vec3 gl,vec3 fY,vec3 fU,vec3 fV,float fW,float fX,vec3 fZ,vec3 fj,vec3 fk,float s,float du,float gc,float gd){vec3 gw=gj-gl;float cv=length(gw);vec3 w=2.0*dot(fZ,fY)*fY-fZ;vec3 gx=dot(gw,w)*w-gw;vec3 gy=gw+gx*clamp(fT/length(gx),0.0,1.0);vec3 R=normalize(gy);float er=clamp(dot(fY,R),0.0,1.0);if(er>0.0){float gm=pow(clamp(1.0-pow(cv/gk,4.0),0.0,1.0),2.0)/(cv*cv+1.0);vec3 bx=fg(fZ,R,R,fY,fW,fX,fV,fU,fj,fk,s,du,gc,gd);float cn=fW*fW;float gz=clamp(cn+fT/(3.0*cv),0.0,1.0);float gA=cn/gz;gA*=gA;return gA*gm*er*bx*fS;}return vec3(0.0);}
#if 0
vec3 gB(vec3 gj,vec3 gC,vec3 gD,vec3 gE,float gF,float gG,float gk,vec3 fS,vec3 gl,vec3 fY,vec3 fU,vec3 fV,float fW,float fX,vec3 fZ,float s,float du){vec3 gH=cI(gl,gj,gC);vec3 cu=gH-gj;vec2 gI=vec2(dot(cu,gE),dot(cu,gD));vec2 gJ=vec2(clamp(gI.x,-gF,gF),clamp(gI.y,-gG,gG));vec3 gK=gj+(gE*gJ.x+gD*gJ.y);vec3 R=gK-gl;float cv=length(R);R/=cv;float er=max(dot(gC,-R),0.0);if(er>0.0&&cM(gl,gj,gC)==1){float gm=pow(clamp(1.0-pow(cv/gk,4.0),0.0,1.0),2.0)/(cv*cv+1.0);vec3 dU=reflect(fZ,fY);vec3 gL=cP(gl,dU,gj,gC);float gM=dot(dU,gC);vec3 fh=vec3(0.0);vec3 bx=vec3(0.0);if(gM>0.0){vec3 gN=gL-gj;vec2 gO=vec2(dot(gN,gE),dot(gN,gD));vec2 gP=vec2(clamp(gO.x,-gF,gF),clamp(gO.y,-gG,gG));vec3 gQ=gj+(gE*gP.x+gD*gP.y);vec3 gR=normalize(gQ-gl);vec3 gS=normalize(gR+fZ);vec3 bx=getFresnel(fV,fZ,gS);fh=ggxSpecular(fZ,gR,fY,gS,fW)*bx;}float gT=max(dot(fY,R),0.0);vec3 cW=(vec3(1.0)-bx)*fU*INV_PI*sqrt(er*gT);return gm*gM*(cW+fh)*fS;}return vec3(0.0);}
#endif
vec2 gU(vec3 k,vec2 gV,vec2 gW){vec2 dR=vec2(k.x/(1.0-k.z),k.y/(1.0-k.z));dR=gV*gW.y+dR*gW.y;return dR;}float gX(float gY,float gZ,float ha){float hb=clamp(gY/gZ*ha,0.0,ha);return mix(hb,ha,ha);}float hc(float bx){float hd=sqrt(bx);return (hd+1.0)/(1.0-hd);}
#if defined(USE_AMBIENT_IBL)

#if defined(USE_CUBEMAP_FORMAT)
vec3 he(samplerCube hf,vec3 ga,float hg){
#if defined(USE_LOGLUV_INPUT)
return cf(W(hf,ga,hg));
#elif defined(USE_RGBM_RADIANCE_PROBE_INPUT)
return a(cj(W(hf,ga,hg),u_rgbmRadianceProbeMaxRange));
#else
return W(hf,ga,hg).rgb;
#endif
}
#else
vec3 he(sampler2D hf,vec3 ga,float hg){
#if defined(USE_LOGLUV_INPUT)
return cf(W(hf,ga,hg));
#elif defined(USE_RGBM_RADIANCE_PROBE_INPUT)
return a(cj(W(hf,ga,hg),u_rgbmRadianceProbeMaxRange));
#else
return W(hf,ga,hg).rgb;
#endif
}
#endif
float hh(float fW){
#if defined(USE_SPECULAR_POWER_IBL_LOOKUP)
float bp=bn(fW);return max(0.0,bo(bp,u_iblConvData.x,u_iblConvData.y,u_iblConvData.z));
#else
return sqrt(fW)*u_radianceMipLevels;
#endif
}
#if defined(USE_ENVIRONMENT_REFRACTION)

#if defined(USE_ENVIRONMENT_REFRACTION_DISPERSION)
float hi(float hj){float hk=hj*hj;return sqrt(0.3306*hk/(hk-175.0*175.0)+4.3356*hk/(hk-106.0*106.0));}
#endif
vec3 hl(vec3 fL,vec3 hm,vec3 fZ,vec3 q,float r,samplerCube hn,samplerCube ho,mat4 hp,vec2 dS,float hj){float hq=0.0;
#if defined(USE_ENVIRONMENT_REFRACTION_DISPERSION)
float bG=hi(hj);
#else
float bG=hc(q.x);
#endif
vec3 w=refract(-fZ,hm,1.0/bG);float hr=1.0-dq(q.x,dot(fZ,hm));vec3 hs=dT(fL,w,ho,dS,hq);vec3 m=vec3(0.0);float ht=r;for(int ef=0;ef<10;++ef){vec3 hu=textureCube(hn,hs).xyz*2.0-1.0;vec3 hv=refract(w,hu,bG);float hw=dot(-w,hu);float hx=hr;if(hw>0.01){hx=max(0.0,1.0-hx*dq(q.x,hw));}vec3 hy=dT(hs,hv,ho,dS,hq);m+=hx*he(u_radianceTexture,hy,hh(ht));ht+=r;hr-=hx;if(hr<=0.001){break;}fL=fL+w*dP(ho,hs,dS);w=reflect(w,hu);hs=dT(fL,w,ho,dS,hq);}return m;}
#endif
vec3 hz(vec3 hA,vec3 gl,vec3 fY,vec3 fV,vec3 fU,float fW,float fX,vec3 fZ,vec3 hB,float hC,float hD,vec4 bJ,mat3 hE,vec3 fj,vec3 hF,float s,float du,vec3 hG,float gc,float gd){
#if defined(USE_ANISOTROPIC_BRDF)
vec3 hH=cross(fZ,hF);vec3 hI=cross(hH,hF);vec3 hJ=normalize(mix(fY,hI,fX));hB=-reflect(fZ,hJ);
#endif
float hK=fW*fW;
#if defined(USE_PARALLAX_CORRECTION)
vec3 hL=mix(fY,hB,(1.0-hK)*(sqrt(1.0-hK)+hK));vec3 hM=(u_localizedEnvironmentInverseMatrix*vec4(gl,1.0)).xyz;vec3 hN=mat3(u_localizedEnvironmentInverseMatrix)*hL;vec2 hO=cZ(hM,hN,u_localizedEnvironmentMin,u_localizedEnvironmentMax);if(hO.y>hO.x){vec3 hP=hM+hO.y*hN;hP=hP-u_localizedEnvironmentOffset;hB=mix(hP,hN,fW);fW=gX(hO.y,length(hP),fW);}vec3 ga=hB;
#else
vec3 ga=mix(fY,hB,(1.0-hK)*(sqrt(1.0-hK)+hK));
#endif
float hg=hh(fW);
#if defined(USE_CLEARCOAT)
float hQ=hh(gd);
#endif
vec3 hR=vec3(0.0);hR=he(u_radianceTexture,ga,hg);
#if defined(USE_CLEARCOAT)
vec3 hS=vec3(0.0);
#if (defined(USE_NORMAL_MAP) && defined(USE_VERTEX_NORMAL))
vec3 hT=normalize(v_normal);vec3 hU=-reflect(fZ,hT);hS=he(u_radianceTexture,hU,hQ);
#else
hS=he(u_radianceTexture,hB,hQ);
#endif
#endif
#if defined(USE_HIGHP_FLOAT)
#if defined(USE_BURLEY_DIFFUSE)
vec3 hV=mix(fY,fZ,clamp((s*(1.02341*rSquared-1.51174)+-0.511705*rSquared+0.755868)*rSquared,0.0,1.0));
#else
vec3 hV=fY;
#endif
vec3 hW=vec3(0.0);
#if defined(USE_AMBIENT_DIFFUSE_SH)
hW=hC*INV_PI*fU*dy(u_irradianceShCoefficients,hV);
#elif defined(USE_AMBIENT_DIFFUSE_PROBE_TEXTURE)
#if defined(USE_LOGLUV_INPUT)
hW=hC*fU*cf(W(u_irradianceProbeTexture,hV,0.0));
#elif defined(USE_RGBM_IRRADIANCE_PROBE_INPUT)
hW=hC*fU*a(cj(W(u_irradianceProbeTexture,hV,0.0),u_rgbmIrradianceProbeMaxRange));
#else
hW=hC*fU*W(u_irradianceProbeTexture,hV,0.0).rgb;
#endif
#elif defined(USE_AMBIENT_DIFFUSE_TEXTURE)
hW=hA;
#if !defined(USE_IRRADIANCE_INCLUDING_DIFFUSE)
hW*=fU;
#endif
#if defined(APPLY_AO_ON_IRRADIANCE)
hW*=hC;
#endif
#else
#if defined(USE_LOGLUV_INPUT)
hW=hC*fU*cf(W(u_radianceTexture,hV,u_radianceMipLevels));
#elif defined(USE_RGBM_RADIANCE_PROBE_INPUT)
hW=hC*fU*a(cj(W(u_radianceTexture,hV,u_radianceMipLevels),u_rgbmRadianceProbeMaxRange));
#else
hW=hC*fU*W(u_radianceTexture,hV,u_radianceMipLevels).rgb;
#endif
#endif
#endif
vec3 hX=vec3(0.0);float hY=1.0;
#if defined(USE_ENV_BRDF_LUT)
vec4 hZ=texture2D(u_brdfTexture,vec2(s,fW));float ia=hZ.x;float ib=hZ.y;
#if defined(USE_BURLEY_DIFFUSE)
hY=hZ.z;
#endif
#if defined(USE_CLEARCOAT)
hX=fV*ia+vec3(du*ib*(1.0-gc));
#else
hX=fV*ia+vec3(du*ib);
#endif
#else
#if defined(USE_CLEARCOAT)
hX=p(fV,fW,s,gc);
#else
hX=p(fV,fW,s,0.0);
#endif
#endif
#if defined(USE_AMBIENT_SSR)
vec4 ic=texture2D(u_ssrTraceTexture,gl_FragCoord.xy/u_size);hR=(1.0-ic.w)*hR+ic.w*ic.rgb;
#endif
vec3 id=hX*hR;
#if defined(USE_STATIC_AMBIENT_DIRECTIONAL_OCCLUSION)
id*=bI(hE*ga,bJ);
#else
id*=hD;
#endif
#if defined(USE_HIGHP_FLOAT)
vec3 ie=hY*hW;
#else
vec3 ie=fU;
#endif
#if defined(USE_CLEARCOAT)
#if (defined(USE_NORMAL_MAP) && defined(USE_VERTEX_NORMAL))
float ig=clamp(dot(hT,fZ),0.0,1.0);float ih=dq(0.04,ig);
#else
float ih=dq(0.04,s);
#endif
ih*=gc;float fu=(1.0-ih);ie*=fu;id*=fu;id+=hS*ih;
#endif
vec3 ii=vec3(0.0);
#if defined(USE_ENVIRONMENT_REFRACTION)
#if defined(USE_ENVIRONMENT_REFRACTION_DISPERSION)
ii.r+=hG.r*hl(gl,hV,fZ,fV,fW,u_environmentRefractionNormalTexture,u_environmentRefractionDepthTexture,u_environmentRefractionMatrix,u_environmentRefractionNearFar,650.0).r;ii.g+=hG.g*hl(gl,hV,fZ,fV,fW,u_environmentRefractionNormalTexture,u_environmentRefractionDepthTexture,u_environmentRefractionMatrix,u_environmentRefractionNearFar,510.0).g;ii.b+=hG.b*hl(gl,hV,fZ,fV,fW,u_environmentRefractionNormalTexture,u_environmentRefractionDepthTexture,u_environmentRefractionMatrix,u_environmentRefractionNearFar,441.0).b;
#else
ii+=hG*hl(gl,hV,fZ,fV,fW,u_environmentRefractionNormalTexture,u_environmentRefractionDepthTexture,u_environmentRefractionMatrix,u_environmentRefractionNearFar,0.0);
#endif
#endif
return ie+id+ii;}
#else
vec3 hz(vec3 hA,vec3 fU){vec3 hW=hA;
#if !defined(USE_IRRADIANCE_INCLUDING_DIFFUSE)
hW*=fU;
#endif
return hW;}
#endif
float ij(float z,float en){return pow(en+0.01,4.0)+max(1e-2,min(3.0*1e3,100.0/(1e-5+pow(abs(z)/10.0,3.0)+pow(abs(z)/200.0,6.0))));}void main(){vec3 m=vec3(0.0);
#if (defined(USE_CUBE_IBR) || defined(USE_LATLONG_IBR))
#if defined(USE_DEFERRED_TEXTURING)
#if defined(USE_CUBE_IBR)
float ik=textureCube(u_materialIdTexture,v_direction).r;vec4 il=textureCube(u_uvTexture,v_direction);
#elif
float ik=texture2D(u_materialIdTexture,Q(v_direction)).r;vec4 il=texture2D(u_uvTexture,Q(v_direction));
#endif
vec2 bP=vec2(by(il.xy),by(il.zw));float im=floor(256.0*min(ik,0.9999));vec2 ik=vec2(floor(mod(im,16.0)),floor(im/16.0));vec2 texCoord0=bP/vec2(16.0);texCoord0+=vec2(ik.x/16.0,(16.0-1.0-ik.y)/16.0);
#else
vec3 texCoord0=v_direction;
#endif
#elif (defined(LIT) || defined(LIT_PARTICLE))
vec2 texCoord0=v_texCoord0;
#elif defined(LIT_POINT_SPRITE)
vec2 texCoord0=gl_PointCoord;
#endif
#if defined(USE_2ND_UV_FOR_BAKED_MAPS)
vec2 bakedTc=v_texCoord1;
#else
SAMPLER_COORD_TYPE bakedTc=texCoord0;
#endif
#if (defined(USE_OIT_TRANSPARENCY) || defined(USE_TRANSPARENCY) || defined(USE_REFRACTION))
float io=float(OPACITY_DEFAULT);
#endif
vec3 eS=vec3(DIFFUSE_DEFAULT);
#if defined(USE_DIFFUSE_TEXTURE_WITH_ALPHA)
vec4 ip=vec4(0.0);
#if defined(USE_GAMMA_SPACE)
ip=f(SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD));
#else
ip=SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD);
#endif
eS=ip.rgb;io=ip.a;
#elif defined(USE_DIFFUSE_TEXTURE_WITH_OCCLUSION)
vec4 iq=vec4(0.0);iq=SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD);
#if defined(USE_GAMMA_SPACE)
eS=a(iq.rgb);
#else
eS=iq.rgb;
#endif
#elif defined(USE_DIFFUSE_TEXTURE_WITH_METALLIC)
vec4 ir=vec4(0.0);ir=SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD);
#if defined(USE_GAMMA_SPACE)
eS=a(ir.rgb);
#else
eS=ir.rgb;
#endif
#elif defined(USE_DIFFUSE_TEXTURE)
#if defined(USE_RGBM_DIFFUSE_INPUT)
eS=a(cj(SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD),u_rgbmDiffuseMaxRange));
#else
#if defined(USE_GAMMA_SPACE)
eS=f(SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD)).rgb;
#else
eS=SAMPLER_LOOKUP(u_diffuseTexture,SAMPLER_TEXCOORD).rgb;
#endif
#endif
#elif defined(USE_DIFFUSE_COLOR)
#if defined(USE_GAMMA_SPACE)
eS=a(u_diffuseColor);
#else
eS=u_diffuseColor;
#endif
#endif
#if defined(USE_DIFFUSE_SPRITE_ANIMATION)
vec4 is=texture2D(u_diffuseSpriteAnimationTexture,v_texCoord2);
#if defined(USE_GAMMA_SPACE)
is=f(is);
#endif
eS=mix(eS,is.rgb,is.a*u_diffuseSpriteAnimationAlpha);
#endif
#if defined(USE_HDR)
#if defined(USE_HIGH_LUMINANCE_DIFFUSE_BOOST)
m+=eS*step(u_highLuminanceDiffuseBoostParams.x,l(eS))*u_highLuminanceDiffuseBoostParams.y;
#endif
#endif
#if (defined(USE_OIT_TRANSPARENCY) || defined(USE_TRANSPARENCY))
#if defined(USE_OPACITY_TEXTURE)
io*=SAMPLER_LOOKUP(u_opacityTexture,SAMPLER_TEXCOORD).r;
#elif defined(USE_OPACITY_VALUE)
io*=u_opacity;
#endif
#endif
vec3 hm=vec3(1.0);
#if defined(USE_VERTEX_NORMAL)
hm=normalize(v_normal);
#else
#if defined(LIT_POINT_SPRITE)
hm.xy=(texCoord0*vec2(2.0,-2.0)+vec2(-1.0,1.0))*0.2;float it=dot(hm.xy,hm.xy);if(it>0.04) discard;
#if defined(USE_TRANSPARENCY)
io=1.0-it;
#endif
hm.z=sqrt(1.0-it);
#else
#extension GL_OES_standard_derivatives : enable
#if defined(USE_RIGHT_HANDED_COORDINATE_SYSTEM)
hm=normalize(cross(dFdx(v_position),dFdy(v_position)));
#else
hm=normalize(cross(dFdy(v_position),dFdx(v_position)));
#endif
#endif
#endif
#if defined(LIT_POINT_SPRITE)
#if (defined(HAS_SHADOWED_LIGHTS))
vec4 iu=u_projectionMatrix*vec4(hm,1.0);vec3 iv=iu.xyz/iu.w+v_positionCS;
#endif
hm=normalize(vec3(u_inverseViewMatrix*vec4(hm,1.0)).xyz);vec3 iw=v_radius*hm+v_position;
#else
vec3 iw=v_position;
#if (defined(HAS_SHADOWED_LIGHTS))
vec3 iv=v_positionCS;
#endif
#endif
vec3 fZ=normalize(u_cameraLocation-iw);float r=ROUGHNESS_DEFAULT;
#if defined(USE_ROUGHNESS_TEXTURE)
r=SAMPLER_LOOKUP(u_roughnessTexture,SAMPLER_TEXCOORD).r;
#elif defined(USE_ROUGHNESS_VALUE)
r=u_roughness;
#endif
float ix=0.0;
#if defined(USE_AO_WITH_WETNESS)
vec2 iy=SAMPLER_LOOKUP(u_aoTextureStatic,SAMPLER_BAKED_TEXCOORD).rg;ix=iy.g;
#elif defined(USE_WETNESS_TEXTURE)
ix=SAMPLER_LOOKUP(u_wetnessTexture,SAMPLER_BAKED_TEXCOORD).r;
#elif defined(USE_WETNESS_VALUE)
ix=u_wetness;
#endif
vec3 hG=vec3(0.0);
#if (defined(USE_ENVIRONMENT_REFRACTION) || defined(USE_REFRACTION))
hG=eS;eS*=r;
#endif
mat3 hE=mat3(0.0);vec3 fj=vec3(0.0);vec3 hF=vec3(0.0);
#if (defined(USE_NORMAL_MAP) || defined(USE_STATIC_AMBIENT_DIRECTIONAL_OCCLUSION) || defined(USE_ANISOTROPIC_BRDF) || defined(USE_BENT_NORMAL_MAP))
#if defined(USE_VERTEX_TANGENT_SPACE)
fj=normalize(v_tangent);hF=normalize(v_bitangent);mat3 iz=mat3(fj,hF,hm);
#else
mat3 iz=bM(hm,iw,texCoord0);fj=iz[0];hF=iz[1];
#endif
hE=n(iz);
#endif
#if (defined(USE_NORMAL_MAP) || defined(USE_STATIC_AMBIENT_DIRECTIONAL_OCCLUSION))
#if defined(USE_NORMAL_MAP)
vec3 iA=vec3(1.0);
#if defined(USE_NORMAL_MAP_WITH_ROUGHNESS)
vec4 iB=SAMPLER_LOOKUP(u_normalTexture,SAMPLER_TEXCOORD);
#if defined(USE_NORMAL_MAP_RABG)
iA.xy=iB.ag*2.0-1.0;iA.z=sqrt(1.0-dot(iA.xy,iA.xy));r=iB.r;
#else
iA=iB.xyz*2.0-1.0;r=iB.w;
#endif
#else
#if defined(USE_RGBM_NORMAL_INPUT)
iA=a(cj(SAMPLER_LOOKUP(u_normalTexture,SAMPLER_TEXCOORD),u_rgbmNormalMaxRange))*2.0-1.0;
#else
iA=SAMPLER_LOOKUP(u_normalTexture,SAMPLER_TEXCOORD).xyz*2.0-1.0;
#endif
#endif
hm=iz*mix(vec3(0.0,0.0,1.0),iA,u_normalMapStrength);
#endif
#endif
#if defined(USE_TWO_SIDED_MATERIAL)
hm*=sign(dot(hm,fZ));
#endif
#if defined(USE_GAMMA_SPACE_ROUGHNESS)
r=pow(r,2.2);
#endif
#if defined(USE_GLOSSINESS_WORKFLOW)
r=1.0-r;
#elif defined(USE_SPECULAR_POWER_WORKFLOW)
r=sqrt(2.0/(r+1.0));
#endif
float s=clamp(dot(hm,fZ),0.0,1.0);float fi=0.0;
#if defined(USE_ANISOTROPY_TEXTURE)
fi=SAMPLER_LOOKUP(u_anisotropyTexture,SAMPLER_TEXCOORD).r;
#elif defined(USE_ANISOTROPY_VALUE)
fi=u_anisotropy;
#endif
vec3 q=vec3(SPECULAR_DEFAULT);
#if defined(USE_METALLIC_WORKFLOW)
float iC=0.0;
#endif
#if defined(USE_SPECULAR_TEXTURE)
#if defined(USE_REFLECTANCE_WITH_ANISOTROPY)
vec4 iD=SAMPLER_LOOKUP(u_specularTexture,SAMPLER_TEXCOORD);fi=iD.a;q=iD.rgb;
#else
q=SAMPLER_LOOKUP(u_specularTexture,SAMPLER_TEXCOORD).rgb;
#endif
#if defined(USE_GAMMA_SPACE_SPECULAR)
q=a(q);
#endif
#elif defined(USE_SPECULAR_COLOR)
#if defined(USE_GAMMA_SPACE_SPECULAR)
q=a(u_specularColor);
#else
q=u_specularColor;
#endif
#elif defined(USE_METALLIC_TEXTURE)
#if defined(USE_REFLECTANCE_WITH_ANISOTROPY)
metallicAnisotropy=SAMPLER_LOOKUP(u_metallicTexture,SAMPLER_TEXCOORD).xy;iC=metallicAnisotropy.x;fi=metallicAnisotropy.y;
#else
iC=SAMPLER_LOOKUP(u_metallicTexture,SAMPLER_TEXCOORD).x;
#endif
#elif defined(USE_METALLIC_VALUE)
iC=u_metallicValue;
#elif defined(USE_DIFFUSE_TEXTURE_WITH_METALLIC)
iC=ir.a;
#endif
float t=0.0;
#if defined(USE_CLEARCOAT_TEXTURE)
t=SAMPLER_LOOKUP(u_clearcoatTexture,SAMPLER_TEXCOORD).x;
#elif defined(USE_CLEARCOAT_VALUE)
t=u_clearcoatValue;
#endif
float fl=0.0;
#if defined(USE_CLEARCOAT)
fl=u_clearcoatRoughness;
#endif
vec3 hA=vec3(0.0);
#if defined(USE_AMBIENT_DIFFUSE_TEXTURE)
#if defined(USE_IRRADIANCE_WITH_METALLIC)
vec4 iE=SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD);
#if defined(USE_IRRADIANCE_BLEND)
vec4 iF=SAMPLER_LOOKUP(u_irradianceTextureBlendDst,SAMPLER_BAKED_TEXCOORD);
#endif
#if defined(USE_RGBM_IRRADIANCE_INPUT)
hA=a(cl(iE,u_rgbmIrradianceMaxRange,iC));
#if defined(USE_IRRADIANCE_BLEND)
irradianceDst=a(cl(iF,u_rgbmIrradianceMaxRangeBlendDst,iC));
#endif
#else
#if defined(USE_IRRADIANCE_BLEND)
hA=iE.rgb*(1.0-u_irradianceTextureBlendFactor)+iF.rgb*u_irradianceTextureBlendFactor;iC=iE.a*(1.0-u_irradianceTextureBlendFactor)+iF.a*u_irradianceTextureBlendFactor;
#else
hA=iE.rgb;iC=iE.a;
#endif
#endif
#else
#if defined(USE_RGBM_IRRADIANCE_INPUT)
#if defined(USE_IRRADIANCE_BLEND)
hA=a(cj(SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD),u_rgbmIrradianceMaxRange))*(1.0-u_irradianceTextureBlendFactor)+a(cj(SAMPLER_LOOKUP(u_irradianceTextureBlendDst,SAMPLER_BAKED_TEXCOORD),u_rgbmIrradianceMaxRangeBlendDst))*u_irradianceTextureBlendFactor;
#else
hA=a(cj(SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD),u_rgbmIrradianceMaxRange));
#endif
#else
#if defined(USE_GAMMA_SPACE_IRRADIANCE)
#if defined(USE_IRRADIANCE_BLEND)
hA=a(SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD).rgb)*(1.0-u_irradianceTextureBlendFactor)+a(SAMPLER_LOOKUP(u_irradianceTextureBlendDst,SAMPLER_BAKED_TEXCOORD).rgb)*u_irradianceTextureBlendFactor;
#else
hA=a(SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD).rgb);
#endif
#else
#if defined(USE_IRRADIANCE_BLEND)
hA=SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD).rgb*(1.0-u_irradianceTextureBlendFactor)+SAMPLER_LOOKUP(u_irradianceTextureBlendDst,SAMPLER_BAKED_TEXCOORD).rgb*u_irradianceTextureBlendFactor;
#else
hA=SAMPLER_LOOKUP(u_irradianceTexture,SAMPLER_BAKED_TEXCOORD).rgb;
#endif
#endif
#endif
#endif
hA*=u_irradianceTextureBoost;
#endif
#if defined(USE_METALLIC_WORKFLOW)
q=mix(vec3(DIELECTRIC_REFLECTANCE),eS,iC);eS*=(1.0-iC);
#endif
#if (defined(USE_WETNESS_TEXTURE) || defined(USE_WETNESS_VALUE) || defined(USE_AO_WITH_WETNESS))
eS*=mix(1.0,0.3,ix);r*=mix(1.0,0.0,ix);q=mix(q,0.02,ix*ix);
#if defined(USE_NORMAL_MAP)
hm=mix(hm,v_normal,ix*ix);
#endif
#endif
float du=dw(q);
#if (defined(USE_OIT_TRANSPARENCY) || defined(USE_TRANSPARENCY) || defined(USE_REFRACTION))
#if defined(USE_FRESNEL_OPACITY)
io=min(1.0,io+dv(q.g,du,s));
#endif
#if defined(USE_UV_BASED_RADIAL_OPACITY)
io*=clamp(5.0-distance(vec2(0.5),v_originalTexCoord0),0.0,1.0);
#endif
float iG=1.0-io;
#if defined(DISCARD_TRANSPARENT)
if(iG>0.0){discard;}
#elif defined(DISCARD_OPAQUE)
if(iG==0.0){discard;}
#endif
#endif
#if defined(USE_EMISSIVE_TEXTURE)
#if defined(USE_RGBM_EMISSIVE_INPUT)
m+=a(cj(SAMPLER_LOOKUP(u_emissiveTexture,SAMPLER_TEXCOORD)),u_rgbmEmissiveMaxRange)*u_emissiveStrength;
#else
vec4 iH=SAMPLER_LOOKUP(u_emissiveTexture,SAMPLER_TEXCOORD);
#if defined(USE_GAMMA_SPACE_EMISSIVE)
m+=a(iH.rgb)*iH.a*u_emissiveStrength;
#else
m+=iH.rgb*iH.a*u_emissiveStrength;
#endif
#endif
#elif defined(USE_EMISSIVE_COLOR)
#if defined(USE_GAMMA_SPACE_EMISSIVE)
m+=a(u_emissiveColor)*u_emissiveStrength;
#else
m+=u_emissiveColor*u_emissiveStrength;
#endif
#endif
#if defined(USE_HIGHLIGHT_COLOR)
vec2 iI=u_highlightPosT.xy-v_texCoord0.xy;float iJ=dot(iI,iI);float iK=smoothstep(min(u_highlightPosT.z-0.4,0.0),u_highlightPosT.z,iJ)*smoothstep(u_highlightPosT.z+0.4,u_highlightPosT.z,iJ)*(1.0-(1.0-u_highlightPosT.z)*(1.0-u_highlightPosT.z));m=mix(m,u_highlightColor.xyz,iK*u_highlightColor.w);
#endif
vec3 ga=-reflect(fZ,hm);
#if DIR_LIGHT_COUNT > 0
for(int ef=0;ef<DIR_LIGHT_COUNT;++ef){m+=fQ(normalize(-u_directionalLightDirection[ef]),u_directionalLightColor[ef],u_directionalLightRadius[ef],eS,q,r,fi,hm,fZ,ga,1.0,fj,hF,s,du,t,fl);}
#endif
#if SHADOWED_DIR_LIGHT_COUNT > 0
float fq=1.0;for(int ef=0;ef<SHADOWED_DIR_LIGHT_COUNT;++ef){fq=fJ(u_shadowedDirectionalLightShadowMap[ef],u_shadowedDirectionalLightShadowMapSize[ef],u_shadowedDirectionalLightShadowMapViewProjectionMatrix[ef],iw,iv.z);m+=fQ(normalize(-u_shadowedDirectionalLightDirection[ef]),u_shadowedDirectionalLightColor[ef],u_shadowedDirectionalLightRadius[ef],eS,q,r,fi,hm,fZ,ga,fq,fj,hF,s,du,t,fl);}
#endif
#if POINT_LIGHT_COUNT > 0
for(int ef=0;ef<POINT_LIGHT_COUNT;++ef){m+=gi(u_pointLightPosition[ef],u_pointLightFalloff[ef],u_pointLightColor[ef],iw,hm,eS,q,r,fi,fZ,fj,hF,s,du,t,fl);}
#endif
#if SPOT_LIGHT_COUNT > 0
for(int ef=0;ef<SPOT_LIGHT_COUNT;++ef){m+=gn(u_spotLightPosition[ef],-u_spotLightDirection[ef],u_spotLightAngle[ef],u_spotLightFalloff[ef],u_spotLightExponent[ef],u_spotLightColor[ef],v_spotLightProjPosition[ef],u_spotLightGoboTexture[ef],u_spotLightHasGobo[ef],iw,hm,eS,q,r,fi,fZ,fj,hF,s,du,t,fl);}
#endif
#if SPHERE_LIGHT_COUNT > 0
for(int ef=0;ef<SPHERE_LIGHT_COUNT;++ef){m+=gv(u_spherelightPosition[ef],u_sphereLightRadius[ef],u_sphereLightFalloff[ef],u_sphereLightColor[ef],iw,hm,eS,q,r,fi,fZ,fj,hF,s,du,t,fl);}
#endif
#if AREA_LIGHT_COUNT > 0
for(int ef=0;ef<AREA_LIGHT_COUNT;++ef){m+=gB(u_areaLightPosition[ef],u_areaLightNormal[ef],u_areaLightUp[ef],u_areaLightRight[ef],u_areaLightSize[ef].x,u_areaLightSize[ef].y,u_areaLightColor[ef],iw,hm,eS,q,r,fi,fZ,fj,hF,s,du,t,fl);}
#endif
float hC=1.0;float hD=1.0;
#if defined(USE_AMBIENT_IBL)
vec4 bJ=vec4(0.0);
#if defined(USE_STATIC_AMBIENT_OCCLUSION)
#if defined(USE_BENT_NORMAL_MAP)
hm=iz*(SAMPLER_LOOKUP(u_bentNormalTexture,SAMPLER_BAKED_TEXCOORD).xyz*2.0-1.0);s=clamp(dot(hm,fZ),0.0,1.0);
#endif
#if defined(USE_STATIC_AMBIENT_DIRECTIONAL_OCCLUSION)
bJ=SAMPLER_LOOKUP(u_aoTextureStatic,SAMPLER_BAKED_TEXCOORD);hC=bI(hE*hm,bJ);
#else
#if defined(USE_DIFFUSE_TEXTURE_WITH_OCCLUSION)
hC=iq.w;
#else
#if defined(USE_VERTEX_AO)
hC=v_ao.g;
#else
#if defined(USE_AO_WITH_WETNESS)
hC=aoWetnes.r;
#else
hC=SAMPLER_LOOKUP(u_aoTextureStatic,SAMPLER_BAKED_TEXCOORD).r;
#endif
#endif
#endif
#if defined(AO_BLEND_SOFT)
hC=hC*0.7+0.1;
#elif defined(AO_BLEND_STRONG)
hC*=hC;
#endif
#endif
#elif defined(USE_DYNAMIC_AMBIENT_OCCLUSION)
vec2 iL=gl_FragCoord.xy/u_size;hC=texture2D(u_aoTextureDynamic,iL).x;
#endif
hD=clamp(pow(s+hC,exp2(-16.0*r-1.0))-1.0+hC,0.0,1.0);
#endif
#if defined(USE_AMBIENT_COLOR)
m+=hC*u_ambientColor*eS;
#endif
#if SHADOWED_DIR_LIGHT_COUNT == 0
float fq=1.0;
#elif defined(USE_IBL_DIRECTIONAL_SHADOWING)
fq+=0.5;fq=min(fq,1.0);
#else
fq=1.0;
#endif
#if defined(USE_AMBIENT_IBL)
m+=fq*hz(hA,iw,hm,q,eS,r,fi,fZ,ga,hC,hD,bJ,hE,fj,hF,s,du,hG,t,fl);
#else
m+=fq*hz(hA,eS);
#endif
#if defined(USE_Z_DEPTH_FOG)
float iM=-v_positionCS.z/2.0;m=mix(m,vec3(0.0),clamp(iM,0.0,1.0));
#endif
#if defined(USE_SURFACE_VIGNETTING)
vec2 iN=abs((v_texCoord0-vec2(0.5))*2.0);m*=1.0-dot(iN,iN);
#endif
#if !defined(USE_HDR)
vec2 cu=(gl_FragCoord.xy/u_size)-vec2(0.5,0.5);float cv=dot(cu,cu);
#if defined(USE_COLOR_GRADIENT)
m=co(m);
#endif
m*=cB(cu)*cx(cv);m=cs(m);
#endif
#if defined(USE_OIT_TRANSPARENCY)
m*=iG;float E=ij(iw.z,iG);gl_FragColor[0]=vec4(m*E,iG);gl_FragColor[1].r=iG*E;
#elif defined(USE_TRANSPARENCY)
#if defined(USE_REFRACTION)
vec3 iO=refract(-fZ,hm,1.0/hc(q.x));vec3 iP=iw+iO*0.01;vec4 iQ=(u_projectionMatrix*u_viewMatrix*vec4(iP,1.0));iQ.xy/=iQ.w;iQ.xy+=1.0;iQ.xy*=0.5;iQ.x=(iQ.x+max(u_rightEye,0.0))*(0.5+0.5*(1.0-clamp(u_rightEye+1.0,0.0,1.0)));
#if defined(USE_HDR)
m+=hG*texture2D(u_refractionTexture,iQ.xy).rgb*iG;
#else
m=cs(a(m)+hG*a(texture2D(u_refractionTexture,iQ.xy).rgb)*iG);
#endif
gl_FragColor=vec4(m,1.0);
#else
gl_FragColor=vec4(m,io);
#endif
#elif defined(WRITE_DEPTH)
gl_FragColor=vec4(m,iw.z);
#else
gl_FragColor=vec4(m,1.0);
#endif
}
#elif defined(UNLIT)

#if defined(USE_UNLIT_TEXTURE)
uniform sampler2D u_baseTexture;
#endif

#if defined(USE_FRESNEL_OUTLINE)
varying vec3 v_normal;varying vec3 v_position;uniform vec3 u_cameraLocation;uniform float u_fresnelOutlineStrength;
#endif

#if (defined(USE_UNLIT_TEXTURE) || defined(USE_UNLIT_GRADIENT))
varying vec2 v_texCoord0;
#endif

#if defined(USE_SKYBOX)
varying vec3 v_direction;
#if defined(USE_CUBEMAP_FORMAT)
uniform samplerCube u_skyTexture;
#elif defined(USE_STEREOSCOPIC_FORMAT)

#if defined(USE_STEREOSCOPIC_CUBEMAP)
uniform samplerCube u_skyTexture;
#elif defined(USE_STEREOSCOPIC_FISHEYE_LR)
uniform vec3 u_fisheyeParams;uniform sampler2D u_skyTexture;
#else
uniform sampler2D u_skyTexture;
#endif

#else
uniform sampler2D u_skyTexture;
#endif

#elif defined(USE_SKYBOX_SH)
varying vec3 v_direction;uniform vec3 u_shCoeffs[9];
#endif

#if defined(USE_UNLIT_GRADIENT)
uniform vec4 u_unlitGradientColorA;uniform vec4 u_unlitGradientColorB;uniform vec2 u_unlitGradientDirection;vec4 iR(vec4 en,vec4 iS,vec2 Y,vec2 iT){return mix(en,iS,dot(iT,Y));}
#endif

#if defined(USE_UNLIT_COLOR)
uniform vec4 u_color;
#endif

#if defined(USE_UNLIT_GAIN)
uniform vec3 u_gain;
#endif
void main(){vec4 bK=vec4(1.0);
#if defined(USE_UNLIT_GRADIENT)
bK=iR(u_unlitGradientColorA,u_unlitGradientColorB,u_unlitGradientDirection,v_texCoord0);
#endif
#if defined(USE_UNLIT_TEXTURE)
#if defined(FLIP_Y)
vec2 texCoord0=vec2(v_texCoord0.x,1.0-v_texCoord0.y);
#else
vec2 texCoord0=v_texCoord0;
#endif
#if defined(USE_STEREOSCOPIC_FORMAT)
float iU=max(u_rightEye,0.0);vec2 A=texCoord0;
#if defined(USE_STEREOSCOPIC_FLAT_LR)
A.x*=0.5;A.x+=0.5*iU;
#elif defined(USE_STEREOSCOPIC_FLAT_TB)
A.y*=0.5;A.y+=0.5*iU;
#endif
#if defined(CONVERT_TO_BGRA)
bK*=texture2D(u_baseTexture,A).bgra;
#else
bK*=texture2D(u_baseTexture,A);
#endif
#else
#if defined(CONVERT_TO_BGRA)
bK*=texture2D(u_baseTexture,texCoord0).bgra;
#else
bK*=texture2D(u_baseTexture,texCoord0);
#endif
#endif
#if defined(USE_GAMMA_SPACE)
bK=f(bK);
#endif
#endif
#if defined(USE_SKYBOX)
#if defined(USE_STEREOSCOPIC_FORMAT)
float iU=max(u_rightEye,0.0);
#if defined(USE_STEREOSCOPIC_180_LR_SPHERICAL)
vec3 Y=vec3(v_direction.x,-v_direction.y,-v_direction.z);
#if defined(FLIPPED_LATLONG_Y)
Y.y=-Y.y;
#endif
vec2 iV=Y.xy/(2.*sqrt(pow(Y.x,2.)+pow(Y.y,2.)+pow(Y.z+1.,2.)))+.5;
#if !defined(USE_STEREOSCOPIC_PRE_SEPARATED)
iV.x*=0.5;iV.x+=0.5*iU;
#endif
#if defined(FLIP_Y)
iV.y=1.0-iV.y;
#endif
#if defined(CONVERT_TO_BGRA)
bK=texture2D(u_skyTexture,iV).bgra;
#else
bK=texture2D(u_skyTexture,iV);
#endif
#elif defined(USE_STEREOSCOPIC_360_LR)
vec2 A=Q(v_direction);
#if !defined(USE_STEREOSCOPIC_PRE_SEPARATED)
A.x*=0.5;A.x+=0.5*iU;
#endif
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
#if defined(CONVERT_TO_BGRA)
bK=texture2D(u_skyTexture,A).bgra;
#else
bK=texture2D(u_skyTexture,A);
#endif
#elif defined(USE_STEREOSCOPIC_FISHEYE_LR)
vec3 Y=v_direction;
#if defined(FLIP_Y)
Y.y=-Y.y;
#endif
float S=atan(-Y.y,Y.x);float T=atan(sqrt(Y.x*Y.x+Y.y*Y.y),-Y.z);float iW=u_fisheyeParams.y*T/u_fisheyeParams.x;vec2 A=vec2(0.5+iW/u_fisheyeParams.z*cos(S),0.5+iW*sin(S));float iX=step(0.0,A.x)*step(0.0,A.y)*(1.0-step(1.0,A.x))*(1.0-step(1.0,A.y));A.x*=0.5;A.x+=0.5*iU;
#if defined(CONVERT_TO_BGRA)
bK=iX*texture2D(u_skyTexture,A).bgra;
#else
bK=iX*texture2D(u_skyTexture,A);
#endif
#elif defined(USE_STEREOSCOPIC_180_TB_SPHERICAL)
vec3 Y=vec3(v_direction.x,-v_direction.y,-v_direction.z);
#if defined(FLIPPED_LATLONG_Y)
Y.y=-Y.y;
#endif
vec2 iV=Y.xy/(2.*sqrt(pow(Y.x,2.)+pow(Y.y,2.)+pow(Y.z+1.,2.)))+.5;
#if !defined(USE_STEREOSCOPIC_PRE_SEPARATED)
iV.y*=0.5;iV.y+=0.5*iU;
#endif
#if defined(FLIP_Y)
iV.y=1.0-iV.y;
#endif
#if defined(CONVERT_TO_BGRA)
bK=texture2D(u_skyTexture,iV).bgra;
#else
bK=texture2D(u_skyTexture,iV);
#endif
#elif defined(USE_STEREOSCOPIC_360_TB)
vec2 A=Q(v_direction);
#if !defined(USE_STEREOSCOPIC_PRE_SEPARATED)
A.y*=0.5;A.y+=0.5*iU;
#endif
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
#if defined(CONVERT_TO_BGRA)
bK=texture2D(u_skyTexture,A).bgra;
#else
bK=texture2D(u_skyTexture,A);
#endif
#elif defined(USE_STEREOSCOPIC_180_LR)
vec3 Y=vec3(v_direction.z,v_direction.y,v_direction.x*-1.0000001);vec2 A=Q(Y);float bg=1.0-step(0.5,A.x);
#if defined(USE_STEREOSCOPIC_PRE_SEPARATED)
A.x*=2.0;
#else
A.x+=0.5*iU;
#endif
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
#if defined(CONVERT_TO_BGRA)
bK=texture2D(u_skyTexture,A).bgra*bg;
#else
bK=texture2D(u_skyTexture,A)*bg;
#endif
#elif defined(USE_STEREOSCOPIC_180_TB)
vec3 Y=vec3(v_direction.z,v_direction.y,v_direction.x*-1.0000001);vec2 A=Q(Y);float bg=1.0-step(0.5,A.x);
#if !defined(USE_STEREOSCOPIC_PRE_SEPARATED)
A.y*=0.5;A.y+=0.5*iU;
#endif
A.x*=2.0;
#if defined(FLIP_Y)
A.y=1.0-A.y;
#endif
#if defined(CONVERT_TO_BGRA)
bK=texture2D(u_skyTexture,A).bgra*bg;
#else
bK=texture2D(u_skyTexture,A)*bg;
#endif
#elif defined(USE_STEREOSCOPIC_CUBEMAP)
#if defined(CONVERT_TO_BGRA)
bK=textureCube(u_skyTexture,v_direction).bgra;
#else
bK=textureCube(u_skyTexture,v_direction);
#endif
#endif
#else
#if defined(CONVERT_TO_BGRA)
bK=bf(u_skyTexture,v_direction).bgra;
#else
bK=bf(u_skyTexture,v_direction);
#endif
#endif
#if defined(USE_LOGLUV_INPUT)
bK=vec4(cf(f(bK)),1.0);
#elif defined(USE_RGBM_INPUT)
bK=vec4(a(cj(bK,u_rgbmMaxRange)),1.0);
#elif defined(USE_GAMMA_SPACE)
bK=f(bK);
#endif
#elif defined(USE_SKYBOX_SH)
bK.rgb=dy(u_shCoeffs,v_direction);
#endif
#if defined(USE_UNLIT_GAIN)
bK.rgb+=u_gain;
#endif
#if defined(USE_UNLIT_COLOR)
#if defined(USE_GAMMA_SPACE)
bK*=f(u_color);
#else
bK*=u_color;
#endif
#endif
#if defined(USE_FRESNEL_OUTLINE)
vec3 fZ=normalize(u_cameraLocation-v_position);vec3 hm=normalize(v_normal);float s=clamp(dot(hm,fZ),0.0,1.0);bK.a*=dq(u_fresnelOutlineStrength,s);
#endif
#if (defined(USE_SMOOTH_BORDER) && (defined(USE_UNLIT_TEXTURE) || defined(USE_UNLIT_GRADIENT)))
vec2 iY=vec2(float(SMOOTH_STRENGTH))/u_size;vec2 iZ=smoothstep(vec2(0.0),iY,texCoord0)*(1.0-smoothstep(vec2(1.0)-iY,vec2(1.0),texCoord0));bK.a*=iZ.x*iZ.y;
#endif
#if defined(DISCARD_TRANSPARENT)
if(bK.a<=0.1){discard;}
#endif
#if !defined(USE_HDR)
vec2 cu=(gl_FragCoord.xy/u_size)-vec2(0.5,0.5);float cv=dot(cu,cu);
#if defined(USE_COLOR_GRADIENT)
bK.rgb=co(bK.rgb);
#endif
bK.rgb*=cB(cu)*cx(cv);bK.rgb=cs(bK.rgb);
#endif
gl_FragColor=bK;}
#elif (defined(DEPTH_TEXTURE) || defined(SHADOW_MAP) || defined(VARIANCE_SHADOW_MAP) || defined(PARTICLE_DEPTH))
varying vec2 v_depth;
#if defined(DISCARD_ALPHA)
varying vec2 v_texCoord0;uniform sampler2D u_diffuseTexture;
#endif
void main(){
#if defined(DISCARD_ALPHA)
float cn=texture2D(u_diffuseTexture,v_texCoord0).a;if(cn<ALPHA_THRESHOLD){discard;}
#endif
float ja=v_depth.x/v_depth.y;
#if (defined(DEPTH_TEXTURE) || defined(SHADOW_MAP) || defined(PARTICLE_DEPTH))
#if defined(USE_HDR)
gl_FragColor=vec4(ja,1.0,1.0,1.0);
#else
gl_FragColor=bz(ja);
#endif
#elif defined(VARIANCE_SHADOW_MAP)
#extension GL_OES_standard_derivatives : enable
float jb=dFdx(ja);float jc=dFdy(ja);
#if defined(USE_HDR)
gl_FragColor=vec4(ja,ja*ja+0.25*(jb*jb+jc*jc),0.0,1.0);
#else
gl_FragColor=vec4(bw(ja),bw(ja*ja+0.25*(jb*jb+jc*jc)));
#endif
#endif
}
#elif defined(NOP)
void main(){gl_FragColor=vec4(0.0,0.0,0.0,1.0);}
#elif defined(SSR_PRE)
varying vec2 v_depth;varying vec3 v_normal;uniform vec4 u_projection;void main(){gl_FragColor=vec4(normalize(v_normal),u_projection.y/((v_depth.x/v_depth.y)-u_projection.x));}
#elif defined(USE_LENS_FLARE)
uniform sampler2D u_baseTexture;uniform sampler2D u_occlusionTexture;varying vec2 v_texCoord0;varying float v_distToCenter;uniform vec3 u_intensity;void main(){
#if defined(USE_GAMMA_SPACE)
vec4 m=f(texture2D(u_baseTexture,v_texCoord0));
#else
vec4 m=texture2D(u_baseTexture,v_texCoord0);
#endif
#if defined(USE_FLARE_TEXTURE_OCCLUDE)
vec4 jd=texture2D(u_occlusionTexture,v_texCoord0);m.rgb*=u_intensity;m.a*=v_distToCenter*jd.r;
#else
m.rgb*=u_intensity;m.a*=v_distToCenter;
#endif
#if !defined(USE_HDR)
vec2 cu=(gl_FragCoord.xy/u_size)-vec2(0.5,0.5);float cv=dot(cu,cu);m.rgb*=cB(cu)*cx(cv);m.rgb=cs(m.rgb);
#endif
gl_FragColor=m;}
#endif
