"""CivicFix reproducible geographic screening. Not a validated flood prediction.
Run: python geo_osm_extension.py --dem ../rasters_COP30.tar --osm waterways.geojson
Optional OSM GeoJSON must contain actual OSM waterways; no synthetic geometry is used.
"""
import argparse,tarfile,io,json
from pathlib import Path
import numpy as np, rasterio
from rasterio.io import MemoryFile
from scipy.ndimage import minimum_filter, distance_transform_edt
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--dem',required=True);p.add_argument('--osm');p.add_argument('--out',default='outputs');a=p.parse_args()
out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
with tarfile.open(a.dem) as t: raw=t.extractfile('output_hh.tif').read()
with MemoryFile(raw) as mf,mf.open() as ds:
 z=ds.read(1).astype('float64');transform=ds.transform;crs=ds.crs;bounds=ds.bounds;nodata=ds.nodata
 mask=np.isfinite(z)&((z!=nodata) if nodata is not None else True)
 z[~mask]=np.nan;fill=np.nanmedian(z);v=np.where(mask,z,fill)
 lat=(bounds.bottom+bounds.top)/2
 dx=abs(transform.a)*111320*np.cos(np.deg2rad(lat)) if crs.is_geographic else abs(transform.a)
 dy=abs(transform.e)*111320 if crs.is_geographic else abs(transform.e)
 gy,gx=np.gradient(v,dy,dx);slope=np.degrees(np.arctan(np.hypot(gx,gy)))
 local_rel=v-minimum_filter(v,size=15,mode='nearest')
 # low-lying and flat terrain proxy, NOT a flow-accumulation score
 terrain=100*(0.55*np.clip((15-local_rel)/15,0,1)+0.45*np.clip((12-slope)/12,0,1))
 terrain[~mask]=np.nan
 distance=None;osm_status='Not provided; OSM distance not calculated'
 if a.osm:
  import json
  from rasterio.features import rasterize
  from rasterio.warp import transform_geom
  from shapely.geometry import shape,mapping
  gj=json.load(open(a.osm));geoms=[]
  for feature in gj.get('features',[]):
   if not feature.get('geometry'):continue
   geom=feature['geometry']
   if crs.to_epsg()!=4326: geom=transform_geom('EPSG:4326',crs.to_string(),geom)
   if shape(geom).geom_type in ('LineString','MultiLineString','Polygon','MultiPolygon'):
    geoms.append((geom,1))
  if geoms:
   raster=rasterize(geoms,out_shape=z.shape,transform=transform,fill=0,dtype='uint8',all_touched=True)
   if raster.any():
    distance=distance_transform_edt(raster==0,sampling=(dy,dx))
    osm_status=f'{len(geoms)} geometries read; mapped waterways only (covered/culverted channels may be missing)'
 # illustrative combination, chosen a priori, not statistically calibrated
 score=terrain.copy()
 if distance is not None:score=0.75*terrain+25*np.exp(-distance/100)
 score[~mask]=np.nan
 with rasterio.open(out/'terrain_screening_index.tif','w',driver='GTiff',height=z.shape[0],width=z.shape[1],count=1,dtype='float32',crs=crs,transform=transform,nodata=-9999,compress='deflate') as dst:
  dst.write(np.where(mask,score,-9999).astype('float32'),1)
 extent=(bounds.left,bounds.right,bounds.bottom,bounds.top)
 fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 for ax,arr,title in zip(axs,[z,slope,score],['Elevation (m)','Slope (degrees)','Exploratory terrain index (0–100)']):
  im=ax.imshow(arr,extent=extent,cmap='viridis');ax.set_title(title);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');fig.colorbar(im,ax=ax,shrink=.8)
 fig.suptitle('Istanbul COP30 pilot | not validated flood probability')
 fig.savefig(out/'terrain_screening.png',dpi=160);plt.close(fig)
 metadata={'source':'User-supplied Copernicus GLO-30 DEM','bbox':[bounds.left,bounds.bottom,bounds.right,bounds.top], 'crs':str(crs),'osm':osm_status,'formula':'terrain=100*(0.55*clip((15-local_rel_m)/15)+0.45*clip((12-slope_deg)/12)); with OSM: score=.75*terrain+25*exp(-distance_m/100)', 'caution':'Not flood probability, no flow accumulation or independent Istanbul validation.'}
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 print(json.dumps(metadata,indent=2))
