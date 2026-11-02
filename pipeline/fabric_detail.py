"""Conservative visible-fabric continuation for detected clothed portraits.

The rear pattern is inferred from a cloth patch, never a mirrored face/body photo.
"""
import cv2
import numpy as np
from PIL import Image

def infer_fabric(image):
    rgba=np.asarray(image.convert('RGBA').resize((512,512),Image.Resampling.LANCZOS))
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces=detector.detectMultiScale(cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2GRAY),1.1,4,minSize=(28,28))
    if not len(faces):return None
    fx,fy,fw,fh=max(faces,key=lambda f:int(f[2])*int(f[3]))
    mask=rgba[:,:,3]>220;ys,xs=np.where(mask)
    if not len(xs) or ys.max()-(fy+fh)<2*fh:return None
    top=int(fy+fh*1.2);bottom=int(ys.max())
    band=mask[int(top+(bottom-top)*.6):int(top+(bottom-top)*.9)]
    by,bx=np.where(band)
    if len(bx)<100:return None
    left,right=int(np.percentile(bx,3)),int(np.percentile(bx,97))
    hsv=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2HSV)
    roi=np.zeros(mask.shape,bool);roi[top:bottom,left:right]=True;roi&=mask
    valid=roi&(hsv[:,:,1]>55)&(hsv[:,:,2]>45)
    if valid.sum()<500:return None
    hist=np.bincount((hsv[:,:,0][valid]//5).astype(int),minlength=36);hue=(hist.argmax()+.5)*5
    delta=np.abs(hsv[:,:,0].astype(float)-hue);similar=(np.minimum(delta,180-delta)<9)&(hsv[:,:,1]>45)&mask
    gray=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2GRAY)
    patch_size=max(16,min(36,int(fw*.4)));best=None
    for y in range(int(top+(bottom-top)*.4),bottom-patch_size,8):
        for x in range(left,right-patch_size,8):
            purity=similar[y:y+patch_size,x:x+patch_size].mean()
            if purity<.86:continue
            contrast=float(gray[y:y+patch_size,x:x+patch_size].std())
            score=purity+min(contrast,35)/70
            if best is None or score>best[0]:best=(score,x,y)
    if best is None:return None
    _,x,y=best;w,h=image.size
    patch=image.convert('RGB').crop((round(x*w/512),round(y*h/512),round((x+patch_size)*w/512),round((y+patch_size)*h/512)))
    tiled=Image.new('RGB',image.size)
    # Mirror neighbouring tiles so patch boundaries share identical edge pixels.
    # This avoids a grid of hard seams from illumination differences in the photo.
    for row in range(0,h,patch.height):
        for col in range(0,w,patch.width):
            tile=patch
            if (col//patch.width)%2:tile=tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if (row//patch.height)%2:tile=tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            tiled.paste(tile,(col,row))
    return {'texture':tiled,'bounds':(left/512,1-bottom/512,right/512,1-top/512),
            'report':{'mode':'inferred_fabric_continuation','observed_back':False,'tiling':'mirrored_seam_continuous','patch_pixels':[x,y,patch_size,patch_size],
                      'note':'Repeats a visible cloth patch on the rear torso; face, hair, hands and non-human objects retain native appearance.'}}
