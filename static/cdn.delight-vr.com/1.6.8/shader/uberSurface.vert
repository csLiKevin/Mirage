
#define TWOPI 3.141592654*2.0

#define SQRT_3 1.7320508075688772
vec3 a(vec4 b,float c){return c*b.a*b.rgb;}vec3 d(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}vec4 d(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}vec4 e(vec4 x){return d(((x*34.0)+1.0)*x);}vec4 f(vec4 g){return 1.79284291400159-0.85373472095314*g;}float h(vec3 i){const vec2 j=vec2(1.0/6.0,1.0/3.0);const vec4 k=vec4(0.0,0.5,1.0,2.0);vec3 l=floor(i+dot(i,j.yyy));vec3 m=i-l+dot(l,j.xxx);vec3 n=step(m.yzx,m.xyz);vec3 o=1.0-n;vec3 p=min(n.xyz,o.zxy);vec3 q=max(n.xyz,o.zxy);vec3 r=m-p+j.xxx;vec3 s=m-q+j.yyy;vec3 t=m-k.yyy;l=d(l);vec4 u=e(e(e(l.z+vec4(0.0,p.z,q.z,1.0))+l.y+vec4(0.0,p.y,q.y,1.0))+l.x+vec4(0.0,p.x,q.x,1.0));float v=0.142857142857;vec3 w=v*k.wyz-k.xzx;vec4 x=u-49.0*floor(u*w.z*w.z);vec4 y=floor(x*w.z);vec4 z=floor(x-7.0*y);vec4 A=y*w.x+w.yyyy;vec4 B=z*w.x+w.yyyy;vec4 C=1.0-abs(A)-abs(B);vec4 D=vec4(A.xy,B.xy);vec4 E=vec4(A.zw,B.zw);vec4 F=floor(D)*2.0+1.0;vec4 G=floor(E)*2.0+1.0;vec4 H=-step(C,vec4(0.0));vec4 I=D.xzyw+F.xzyw*H.xxyy;vec4 J=E.xzyw+G.xzyw*H.zzww;vec3 K=vec3(I.xy,C.x);vec3 L=vec3(I.zw,C.y);vec3 M=vec3(J.xy,C.z);vec3 N=vec3(J.zw,C.w);vec4 O=f(vec4(dot(K,K),dot(L,L),dot(M,M),dot(N,N)));K*=O.x;L*=O.y;M*=O.z;N*=O.w;vec4 P=max(0.6-vec4(dot(m,m),dot(r,r),dot(s,s),dot(t,t)),0.0);P=P*P;return 42.0*dot(P*P,vec4(dot(K,m),dot(L,r),dot(M,s),dot(N,t)));}
#if (SHADOWED_DIR_LIGHT_COUNT > 0)

#define HAS_SHADOWED_LIGHTS

#endif
uniform mat4 u_worldMatrix;uniform mat4 u_viewMatrix;uniform mat4 u_inverseViewMatrix;uniform mat4 u_projectionMatrix;uniform mat4 u_inverseProjectionMatrix;attribute vec3 position;
#if defined(USE_PRIMITIVE_INFORMATION)
attribute float vertexIndex;uniform vec2 u_vertexInfo;
#endif

#if (defined(PARTICLE) || defined(LIT_PARTICLE))
attribute vec2 texCoord0;attribute vec4 color0;attribute float ttl;attribute vec2 size;attribute float rotation;varying vec2 v_texCoord0;varying vec4 v_color;
#if defined(LIT_PARTICLE)
varying vec3 v_position;uniform sampler2D u_distanceField;
#endif
uniform float u_aspect;void main(){if(ttl<=0.0){gl_Position=vec4(0,0,-20000,1);return;}v_texCoord0=texCoord0;v_color=color0;
#if defined(LIT_PARTICLE)
vec4 Q=u_projectionMatrix*u_viewMatrix*u_worldMatrix*vec4(position,1.0);float R=texture2D(u_distanceField,(Q.xy/Q.w+vec2(1.0))*0.5).x;float S=sin(rotation*R);float T=cos(rotation*R);vec3 U=vec3(size.x,size.y,0.0);U.xz=vec2(U.x*T,U.x*S);vec4 V=u_worldMatrix*vec4(position.x+U.x,position.y+U.y,position.z+U.z,1.0);v_position=V.xyz;gl_Position=u_projectionMatrix*u_viewMatrix*V;
#else
vec4 W=u_projectionMatrix*u_viewMatrix*u_worldMatrix*vec4(position,1.0);float S=sin(rotation);float T=cos(rotation);vec2 U=vec2(size.x*T-size.y*S,(size.x*S+size.y*T)*u_aspect);W.xy+=U;gl_Position=W;
#endif
}
#elif defined(LIT_POINT_SPRITE)

#if (defined(HAS_SHADOWED_LIGHTS))
varying vec3 v_positionCS;
#endif
varying vec3 v_position;varying float v_radius;attribute float radius;uniform float u_radiusScale;uniform float u_screenWidth;void main(){vec4 V=u_worldMatrix*vec4(position,1.0);v_position=V.xyz;vec4 X=u_viewMatrix*V;float g=u_radiusScale*radius;v_radius=g;vec4 Y=u_projectionMatrix*vec4(g,g,X.z,X.w);
#if (defined(HAS_SHADOWED_LIGHTS))
V=u_projectionMatrix*X;v_positionCS=V.xyz/V.w;
#endif
gl_PointSize=u_screenWidth*Y.x/Y.w;gl_Position=u_projectionMatrix*X;}
#elif defined(LIT)
attribute vec2 texCoord0;varying vec2 v_texCoord0;
#if defined(USE_UV_BASED_RADIAL_OPACITY)
varying vec2 v_originalTexCoord0;
#endif

#if defined(USE_2ND_UV_FOR_BAKED_MAPS)
attribute vec2 texCoord1;varying vec2 v_texCoord1;
#endif

#if defined(USE_DIFFUSE_SPRITE_ANIMATION)
uniform vec3 u_spriteAnimData;uniform vec2 u_texCoordScaleSpriteAnim;uniform vec2 u_texCoordOffsetSpriteAnim;varying vec2 v_texCoord2;
#endif

#if defined(USE_TEXCOORD_OFFSET)
uniform vec2 u_texCoordOffset;
#endif

#if defined(USE_TEXCOORD_SCALE)
uniform vec2 u_texCoordScale;
#endif
varying vec3 v_position;
#if (defined(HAS_SHADOWED_LIGHTS))
varying vec3 v_positionCS;
#endif

#if (defined(USE_VERTEX_NORMAL))
attribute vec3 normal;varying vec3 v_normal;
#if defined(USE_RANDOM_VERTEX_OFFSET)
uniform vec3 u_vertexOffsetParams;
#endif

#if defined(USE_RANDOM_VERTEX_OFFSET)
uniform vec4 u_impactParams;float Z(vec3 i,float ba,float bb,float bc,float bd){float be=max(bb+bc,0.1)*50.0;return h(i)*ba+sin(be)/(be)*bd*(1.0-bc)*(1.0-smoothstep(0.0,bc,bb));}
#endif

#if defined(USE_CONSTANT_RANDOM_VERTEX_OFFSET)
uniform vec3 u_constantVertexOffsetParams;uniform vec3 u_constantVertexOffsetDirection;uniform vec2 u_constantVertexOffsetPosition;
#endif

#endif

#if (defined(USE_LATLONG_IBR) || defined(USE_CUBE_IBR) || defined(USE_VERTEX_DISPLACEMENT))

#define PI 3.141592654

#define INV_PI 0.3183098861

#define INV_TWOPI 0.1591549431
vec2 bf(vec3 g){float o=length(g);float bg=acos(g.y/o);float bh=atan(vec2(g.x),vec2(-g.z)).x;return vec2((PI+bh)*INV_TWOPI,bg*INV_PI);}uniform vec2 u_nearFar;
#if defined(USE_LATLONG_IBR)

#define SAMPLER_LOOKUP texture2D

#define SAMPLER_TEXCOORD worldToLatLong(texCoord0)

#define SAMPLER_TYPE sampler2D

#define SAMPLER_COORD_TYPE vec3
varying vec3 v_direction;
#elif defined(USE_CUBE_IBR)

#define SAMPLER_LOOKUP textureCube

#define SAMPLER_TEXCOORD texCoord0

#define SAMPLER_TYPE samplerCube

#define SAMPLER_COORD_TYPE vec3
varying vec3 v_direction;
#else

#define SAMPLER_LOOKUP texture2D

#define SAMPLER_TEXCOORD texCoord0

#define SAMPLER_TYPE sampler2D

#define SAMPLER_COORD_TYPE vec2

#endif
uniform SAMPLER_TYPE u_depthTexture;float bi(vec4 bj){return dot(bj,vec4(1.0,1.0/255.0,1.0/65025.0,1.0/160581375.0));}float bk(vec3 bl){return dot(bl,vec3(1.0,1.0/255.0,1.0/65025.0));}float bm(SAMPLER_TYPE bn,SAMPLER_COORD_TYPE texCoord0){
#if defined(USE_DEPTH_PACKING)
return u_nearFar.x+bi(SAMPLER_LOOKUP(bn,SAMPLER_TEXCOORD))*(u_nearFar.y-u_nearFar.x);
#elif defined(USE_DEPTH_24_PACKING)
return u_nearFar.x+bk(SAMPLER_LOOKUP(bn,SAMPLER_TEXCOORD).rgb)*(u_nearFar.y-u_nearFar.x);
#else
return u_nearFar.x+SAMPLER_LOOKUP(bn,SAMPLER_TEXCOORD).x*(u_nearFar.y-u_nearFar.x);
#endif
}
#endif

#if defined(USE_VERTEX_AO)
attribute vec4 color0;varying vec3 v_ao;
#endif

#if (defined(USE_VERTEX_TANGENT_SPACE))
attribute vec4 tangent;varying vec3 v_tangent;varying vec3 v_bitangent;
#endif

#if SPOT_LIGHT_COUNT > 0
uniform mat4 u_spotLightViewProjectionMatrix[SPOT_LIGHT_COUNT];varying vec4 v_spotLightProjPosition[SPOT_LIGHT_COUNT];
#endif
void main(){
#if defined(USE_UV_BASED_RADIAL_OPACITY)
v_originalTexCoord0=texCoord0;
#endif
vec2 bo=texCoord0;
#if defined(USE_2ND_UV_FOR_BAKED_MAPS)
v_texCoord1=texCoord1;
#endif
#if defined(USE_TEXCOORD_SCALE)
bo*=u_texCoordScale;
#endif
#if defined(USE_TEXCOORD_OFFSET)
bo+=u_texCoordOffset;
#endif
#if defined(USE_DIFFUSE_SPRITE_ANIMATION)
vec2 bp=clamp((texCoord0-u_texCoordOffsetSpriteAnim)*u_texCoordScaleSpriteAnim,0.0,1.0);float bq=u_spriteAnimData.x*u_spriteAnimData.y;float br=floor(bq*min(u_spriteAnimData.z,0.9999));vec2 bs=vec2(floor(mod(br,u_spriteAnimData.x)),floor(br/u_spriteAnimData.x));bp/=u_spriteAnimData.xy;v_texCoord2=bp+bs/u_spriteAnimData.xy;
#endif
v_texCoord0=bo;
#if defined(USE_VERTEX_AO)
v_ao=color0.rgb;
#endif
vec4 V=vec4(position,1.0);
#if defined(USE_VERTEX_NORMAL)
vec3 bt=normal;
#if defined(USE_RANDOM_VERTEX_OFFSET)
V.xyz+=Z(vec3(v_texCoord0.x,v_texCoord0.y+sin(u_vertexOffsetParams.y*TWOPI),v_texCoord0.y+cos(u_vertexOffsetParams.y*TWOPI))*u_vertexOffsetParams.z,u_vertexOffsetParams.x,distance(u_impactParams.xy,v_texCoord0),u_impactParams.z,u_impactParams.w)*normal;
#endif
#if defined(USE_VERTEX_DISPLACEMENT)
V.xyz+=bm(u_depthTexture,v_texCoord0)*normal;
#endif
#endif
#if defined(USE_CONSTANT_RANDOM_VERTEX_OFFSET)
V.xyz+=Z(vec3(u_constantVertexOffsetPosition.x,u_constantVertexOffsetPosition.y+sin(u_constantVertexOffsetParams.y*TWOPI),u_constantVertexOffsetPosition.y+cos(u_constantVertexOffsetParams.y*TWOPI))*u_constantVertexOffsetParams.z,u_constantVertexOffsetParams.x,distance(u_impactParams.xy,v_texCoord0),u_impactParams.z)*u_constantVertexOffsetDirection;
#endif
#if (defined(USE_LATLONG_IBR) || defined(USE_CUBE_IBR))
V.xyz+=(bm(u_depthTexture,position)-1.0)*position;v_direction=position;
#endif
V=u_worldMatrix*V;
#if SPOT_LIGHT_COUNT > 0
for(int l=0;l<SPOT_LIGHT_COUNT;++l){v_spotLightProjPosition[l]=u_spotLightViewProjectionMatrix[l]*V;}
#endif
v_position=V.xyz;mat3 bu=mat3(u_worldMatrix);
#if defined(USE_VERTEX_NORMAL)
v_normal=bu*bt;
#endif
#if defined(USE_VERTEX_TANGENT_SPACE)
v_tangent=bu*tangent.xyz;v_bitangent=cross(normal,tangent.xyz)*tangent.w;
#endif
#if defined(USE_PARABOLOID_PROJECTION)
V=u_viewMatrix*V;float bv=length(V.xyz);V.xyz/=bv;V.z+=1.0;V.xy/V.z;V.z=(bv-0.1)/(100.0-0.1);
#else
V=u_projectionMatrix*u_viewMatrix*V;
#endif
#if (defined(HAS_SHADOWED_LIGHTS) || defined(USE_Z_DEPTH_FOG))
v_positionCS=V.xyz/V.w;
#endif
gl_Position=V;}
#elif defined(UNLIT)

#if (defined(USE_UNLIT_TEXTURE) || defined(USE_UNLIT_GRADIENT))
attribute vec2 texCoord0;varying vec2 v_texCoord0;
#if defined(USE_TEXCOORD_OFFSET)
uniform vec2 u_texCoordOffset;
#endif

#if defined(USE_TEXCOORD_SCALE)
uniform vec2 u_texCoordScale;
#endif

#if defined(USE_SPRITE_ANIMATION)
uniform vec3 u_spriteAnimData;
#endif

#endif

#if defined(USE_FRESNEL_OUTLINE)
varying vec3 v_normal;varying vec3 v_position;attribute vec3 normal;
#endif

#if defined(USE_BILLBOARD_OFFSET)
uniform vec2 u_billboardOffset;
#endif

#if (defined(USE_SKYBOX) || defined(USE_SKYBOX_SH))
varying vec3 v_direction;
#endif

#if defined(USE_NDC_SCALING)
uniform vec4 u_ndcPositionScale;
#endif

#if defined(USE_NDC_ROTATION)
uniform float u_ndcRotationAngle;
#endif
void main(){
#if (defined(USE_UNLIT_TEXTURE) || defined(USE_UNLIT_GRADIENT))
vec2 bo=texCoord0;
#if defined(USE_SPRITE_ANIMATION)
float bq=u_spriteAnimData.x*u_spriteAnimData.y;float br=floor(bq*min(u_spriteAnimData.z,0.9999));vec2 bs=vec2(floor(mod(br,u_spriteAnimData.x)),floor(br/u_spriteAnimData.x));bo=bo/u_spriteAnimData.xy;
#if defined(USE_SPRITE_ANIMATION_ORIGIN_TOP_LEFT)
bo+=vec2(bs.x/u_spriteAnimData.x,(u_spriteAnimData.y-1.0-bs.y)/u_spriteAnimData.y);
#else
bo+=bs/u_spriteAnimData.xy;
#endif
#endif
#if defined(USE_TEXCOORD_SCALE)
bo*=u_texCoordScale;
#endif
#if defined(USE_TEXCOORD_OFFSET)
bo+=u_texCoordOffset;
#endif
v_texCoord0=bo;
#endif
mat4 bw=u_viewMatrix;
#if defined(LOCK_TO_CAMERA_POSITION)
bw[3].xyz=vec3(0.0);
#endif
#if defined(USE_FULLSCREEN_QUAD)
gl_Position=vec4(position,1.0);
#elif defined(USE_NDC_SCALING)
vec4 V=vec4(position,1.0);
#if defined(USE_NDC_ROTATION)
float bx=cos(u_ndcRotationAngle);float by=sin(u_ndcRotationAngle);V.xy=vec2(V.x*bx-V.y*by,V.y*bx+V.x*by);
#endif
V.xy*=u_ndcPositionScale.zw;V.xy+=u_ndcPositionScale.xy;gl_Position=V;
#elif defined(USE_BILLBOARDING)
mat4 bz=bw*u_worldMatrix;
#if defined(USE_BILLBOARD_OFFSET)
vec4 bA=(vec4(position,1.0)+vec4(bz[3].xyz,0.0));bA.xy+=u_billboardOffset;gl_Position=u_projectionMatrix*bA;
#else
vec4 bB=vec4(position.x*u_worldMatrix[0][0],position.y*u_worldMatrix[1][1],position.z,1.0);gl_Position=u_projectionMatrix*(bB+vec4(bz[3].xyz,0.0));
#endif
#elif (defined(USE_SKYBOX) || defined(USE_SKYBOX_SH))
v_direction=position;float bC=(-u_projectionMatrix[3][2]/(u_projectionMatrix[2][2]-1.0))/SQRT_3;mat4 bD=mat4(1.0);bD[0][0]=bC;bD[1][1]=bC;bD[2][2]=bC;mat4 bE=u_projectionMatrix*bw*bD*u_worldMatrix;gl_Position=bE*vec4(position,1.0);
#else
#if defined(USE_PARABOLOID_PROJECTION)
vec4 V=bw*u_worldMatrix*vec4(position,1.0);float bv=length(V.xyz);V.xyz/=bv;V.z+=1.0;V.xy/V.z;V.z=(bv-0.1)/(100.0-0.1);gl_Position=V;
#else
#if defined(USE_FRESNEL_OUTLINE)
mat3 bu=mat3(u_worldMatrix);v_normal=bu*normal;vec4 V=vec4(position,1.0);v_position=(u_worldMatrix*V).xyz;
#endif
mat4 bE=u_projectionMatrix*bw*u_worldMatrix;gl_Position=bE*vec4(position,1.0);
#endif
#endif
}
#elif (defined(DEPTH_TEXTURE) || defined(SHADOW_MAP) || defined(VARIANCE_SHADOW_MAP))

#if defined(DISCARD_ALPHA)
attribute vec2 texCoord0;varying vec2 v_texCoord0;
#endif
varying vec2 v_depth;void main(){
#if defined(DISCARD_ALPHA)
v_texCoord0=texCoord0;
#endif
mat4 bE=u_projectionMatrix*u_viewMatrix*u_worldMatrix;vec4 V=bE*vec4(position,1.0);v_depth=V.zw;gl_Position=V;}
#elif defined(NOP)
void main(){mat4 bE=u_projectionMatrix*u_viewMatrix*u_worldMatrix;gl_Position=bE*vec4(position,1.0);}
#elif defined(SSR_PRE)
attribute vec3 normal;varying vec3 v_normal;varying vec2 v_depth;void main(){mat4 bz=u_viewMatrix*u_worldMatrix;vec4 V=u_projectionMatrix*bz*vec4(position,1.0);v_normal=mat3(bz)*normal;v_depth=V.zw;gl_Position=V;}
#elif defined(USE_LENS_FLARE)
uniform vec4 u_offsetScaleAspect;
#if !defined(USE_FLARE_TEXTURE_OCCLUDE)
uniform sampler2D u_occlusionTexture;
#endif

#if defined(USE_CUSTOM_ROTATION)
uniform float u_rotation;
#endif
attribute vec2 texCoord0;varying vec2 v_texCoord0;varying float v_distToCenter;void main(){v_texCoord0=texCoord0;vec4 V=u_projectionMatrix*u_viewMatrix*(u_worldMatrix*vec4(0,0,0,1));vec3 bF=V.xyz/V.w;v_distToCenter=1.4/max(1.4,length(vec2(bF.x*u_offsetScaleAspect.w,bF.y)));v_distToCenter*=v_distToCenter;
#if !defined(USE_FLARE_TEXTURE_OCCLUDE)
float bG=0.0;for(int bH=0;bH<8;++bH){for(int l=0;l<8;++l){bG+=texture2D(u_occlusionTexture,vec2(1.0/float(l),1.0/float(bH))).r;}}bG/=64.0;v_distToCenter*=bG*bG;
#endif
if(abs(bF.x)<=2.3&&abs(bF.y)<=2.3){vec2 bI=u_offsetScaleAspect.xy*bF.xy-bF.xy;vec2 U=vec2(position.x,position.y)*u_offsetScaleAspect.z;
#if defined(USE_OFFSET_SCALING)
U*=dot(bI,bI)*0.19+0.27;
#endif
#if defined(USE_CUSTOM_ROTATION)
float S=sin(u_rotation);float T=cos(u_rotation);U=vec2(U.x*T-U.y*S,U.x*S+U.y*T);
#endif
#if defined(USE_AUTO_ROTATE)
float bJ=atan(vec2(bF.y+bI.y),vec2((bF.x+bI.x)*u_offsetScaleAspect.w)).x;float bK=sin(bJ);float bL=cos(bJ);U=vec2(U.x*bL-U.y*bK,U.x*bK+U.y*bL);
#endif
U.y*=u_offsetScaleAspect.w;bF.xy+=U+bI;gl_Position=vec4(bF.xyz*V.w,V.w);}else{gl_Position=vec4(-10.0,-10.0,-10.0,1.0);}}
#endif
