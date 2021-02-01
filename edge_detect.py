import numpy as np
import cv2, time, math
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
from bilateralfilt import bilatfilt
from dog import deroGauss
from tqdm import tqdm
import sys
sys.path.insert(0, 'pybsds')
from skimage.util import img_as_float
from skimage.io import imread
from pybsds.bsds_dataset import BSDSDataset
from pybsds import evaluate_boundaries
import matplotlib
import os

matplotlib.use('Agg')

GT_DIR = os.path.join('contour-data', 'groundTruth')
IMAGE_DIR = os.path.join('contour-data', 'images')
N_THRESHOLDS = 99
#...........................................................................................
def get_edges(I,sd,nang,angles):
	dim = I.shape
	Idog2d = np.zeros((nang,dim[0],dim[1]))
	for i in range(nang):
		dog2d = deroGauss(17,sd,angles[i])
		Idog2dtemp = abs(conv2(I,dog2d,mode='same',boundary='fill'))
		Idog2dtemp[Idog2dtemp<0]=0
		Idog2d[i,:,:] = Idog2dtemp
	return Idog2d
#...........................................................................................
def nonmaxsup(I,gradang):
	dim = I.shape
	Inms = np.zeros(dim)
	xshift = int(np.round(math.cos(gradang*np.pi/180)))
	yshift = int(np.round(math.sin(gradang*np.pi/180)))
	Ipad = np.pad(I,(1,),'constant',constant_values = (0,0))
	for r in range(1,dim[0]+1):
		for c in range(1,dim[1]+1):
			maggrad = [Ipad[r-xshift,c-yshift],Ipad[r,c],Ipad[r+xshift,c+yshift]]
			if Ipad[r,c] == np.max(maggrad):
				Inms[r-1,c-1] = Ipad[r,c]
	return Inms
#...........................................................................................
def calc_sigt(I,threshval):
	M,N = I.shape
	ulim = np.uint8(np.max(I))	
	N1 = np.count_nonzero(I>threshval)
	N2 = np.count_nonzero(I<=threshval)
	w1 = np.float64(N1)/(M*N)
	w2 = np.float64(N2)/(M*N)
	#print N1,N2,w1,w2
	try:
		u1 = np.sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N1 for i in range(threshval+1,ulim))
		u2 = np.sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N2 for i in range(threshval+1))
		uT = u1*w1+u2*w2
		sigt = w1*w2*(u1-u2)**2
		#print u1,u2,uT,sigt
	except:
		return 0
	return sigt
#...........................................................................................
def get_threshold(I):
	max_sigt = 0
	opt_t = 0
	ulim = np.uint8(np.max(I))
	print(ulim)
	for t in range(ulim+1):
		sigt = calc_sigt(I,t)
		#print t, sigt
		if sigt > max_sigt:
			max_sigt = sigt
			opt_t = t
	print('optimal high threshold: ',opt_t)
	return opt_t
	
#...........................................................................................
def threshold(I,uth):
	lth = uth/2.5
	Ith = np.zeros(I.shape)
	Ith[I>=uth] = 255
	Ith[I<lth] = 0
	Ith[np.multiply(I>=lth, I<uth)] = 100
	return Ith
#...........................................................................................
def hysteresis(I):
	r,c = I.shape
	#xshift = int(np.round(math.cos(gradang*np.pi/180)))
	#yshift = int(np.round(math.sin(gradang*np.pi/180)))
	Ipad = np.pad(I,(1,),'edge')
	c255 = np.count_nonzero(Ipad==255)
	imgchange = True
	for i in range(1,r+1):
		for j in range(1,c+1):
			if Ipad[i,j] == 100:
				#if Ipad[i-xshift,j+yshift]==255 or Ipad[i+xshift,j-yshift]==255: 
				if np.count_nonzero(Ipad[r-1:r+1,c-1:c+1]==255)>0:
					Ipad[i,j] = 255
				else:
					Ipad[i,j] = 0
	Ih = Ipad[1:r+1,1:c+1]
	return Ih



def load_gt_boundaries(imname):
    gt_path = os.path.join(GT_DIR, '{}.mat'.format(imname))
    return BSDSDataset.load_boundaries(gt_path)


def load_pred(output_dir, imname):
    pred_path = os.path.join(output_dir, '{}.png'.format(imname))
    return img_as_float(imread(pred_path))


def display_results(ax, f, im_results, threshold_results, overall_result):
    out_keys = ['threshold', 'f1', 'best_f1', 'area_pr']
    out_name = ['threshold', 'overall max F1 score', 'average max F1 score',
                'area_pr']
    for k, n in zip(out_keys, out_name):
        print('{:>20s}: {:<10.6f}'.format(n, getattr(overall_result, k)))
        f.write('{:>20s}: {:<10.6f}\n'.format(n, getattr(overall_result, k)))
    res = np.array(threshold_results)
    recall = res[:, 1]
    precision = res[recall > 0.01, 2]
    recall = recall[recall > 0.01]
    label_str = '{:0.2f}, {:0.2f}, {:0.2f}'.format(
        overall_result.f1, overall_result.best_f1, overall_result.area_pr)
    ax.plot(recall, precision, 'r', lw=2, label=label_str)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')



def get_imlist(name):
    imlist = np.loadtxt('contour-data/{}.imlist'.format(name))
    return imlist.astype(np.int)

def detect_edges(imlist, out_dir):
    for imname in tqdm(imlist):
        I = cv2.imread(os.path.join(IMAGE_DIR, str(imname) + '.jpg'))
        gimg = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
        print('Bilateral filtering...\n')
        gimg = bilatfilt(gimg,17,3,10)
        
        angles = [0,45,90,135]
        nang = len(angles)
        
        print('Calculating Gradient...\n')
        img_edges = get_edges(gimg,2,nang,angles)
        
        for n in range(nang):
            img_edges[n,:,:] = nonmaxsup(img_edges[n,:,:],angles[n])
        print('after nms: ', np.max(img_edges))
        img_edge = np.max(img_edges,axis=0)
        lim = np.uint8(np.max(img_edge))

        print('Calculating Threshold...\n')
        th = get_threshold(gimg)
        the = get_threshold(img_edge)
        
        print('\nThresholding...\n')
        img_edge = threshold(img_edge, the*0.25)
        
        print('Applying Hysteresis...\n')
        img_edge = nonmaxsup(hysteresis(img_edge),90)
        #img_edge = hysteresis(img_edge)
        print(img_edge)
        
        
        out_file_name = os.path.join(out_dir, str(imname) + '.png')
        cv2.imwrite(out_file_name, img_edge)
        
if __name__ == '__main__':
    imset = 'val'
    imlist = get_imlist(imset)
    output_dir = 'output';
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Running detector:')
    detect_edges(imlist,  output_dir) 

    _load_pred = lambda x: load_pred(output_dir, x)
    print('Evaluating:')
    sample_results, threshold_results, overall_result = \
        evaluate_boundaries.pr_evaluation(N_THRESHOLDS, imlist, load_gt_boundaries,
                                          _load_pred, fast=True, progress=tqdm)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    file_name = os.path.join(output_dir + '_out.txt')
    with open(file_name, 'wt') as f:
        display_results(ax, f, sample_results, threshold_results, overall_result)
    fig.savefig(os.path.join(output_dir + '_pr.pdf'), bbox_inches='tight')

