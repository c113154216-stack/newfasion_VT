import sys, glob, trimesh, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams["font.sans-serif"]=["Microsoft JhengHei"]
d=sorted(glob.glob("output/garmentcode/tshirt__*_*-*-*/"))[-1]
g=trimesh.load(glob.glob(d+"*_boxmesh.obj")[0],process=False,force="mesh")
b=trimesh.load("GarmentCode/assets/bodies/mean_all.obj",process=False,force="mesh"); b.vertices*=100
print("garment bbox cm",g.bounds.round(1).tolist()); print("body bbox",b.bounds.round(1).tolist())
fig=plt.figure(figsize=(15,6))
img=plt.imread(glob.glob(d+"*_pattern.png")[0]); ax=fig.add_subplot(1,3,1); ax.imshow(img); ax.axis("off"); ax.set_title("2D 版片")

for k,(el,az,t) in enumerate([(5,-90,"3D 正面"),(5,0,"3D 側面")]):
    ax=fig.add_subplot(1,3,k+2,projection="3d")
    for m,c,a in [(b,"#d9b99b",0.25),(g,"#3b82f6",0.9)]:
        f=m.faces[::max(1,len(m.faces)//8000)]
        pc=Poly3DCollection(m.vertices[f][:,:,[0,2,1]],facecolor=c,edgecolor="none",alpha=a); ax.add_collection3d(pc)
    ax.set_xlim(-50,50); ax.set_ylim(-50,50); ax.set_zlim(80,180); ax.view_init(el,az); ax.set_box_aspect((1,1,1)); ax.axis("off"); ax.set_title(t)
plt.tight_layout(); plt.savefig(d+"preview.png",dpi=90); print(d+"preview.png")
