# MPPP functions



'''

'''

import numpy as np
from planetaryimage import PDS3Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import colour_demosaicing
from PIL import Image
import urllib.request, json 
import os
import cv2
import time
import glob
from multiprocessing import Pool, Manager
from typing import Tuple

from numpy.linalg import inv, norm, det


class image:
    
    '''
    The image class holds all parameters specific to the IMG file
    '''

    def __init__(self, IMG_path ):
        
        self.IMG_path    = IMG_path
        self.filename    = os.path.basename( IMG_path )
        self.label       = PDS3Image.open( IMG_path ).label                  # PDS image header metadata
        self.image       = np.float32( PDS3Image.open( IMG_path ).image )    # three band float-32 image array
        self.mask_image  = np.ones( self.image.shape[:2] )*255               # one band boolian image array
        self.cam         = self.filename[:2]
        self.sol         = int( self.filename[4:8] )
        
        # int to float scaling factor
        self.scale       = self.label['DERIVED_IMAGE_PARMS']['RADIANCE_SCALING_FACTOR'][0]
        self.image      *= self.scale
        
        self.find_offsets_mode = None
        
        try:
            self.site  = int( self.label['ROVER_MOTION_COUNTER'][0] )
            self.drive = int( self.label['ROVER_MOTION_COUNTER'][1] )
            self.LMST  = self.label['LOCAL_MEAN_SOLAR_TIME'].split('M')[1]
        except:
            self.site  = 0
            self.drive = 0
            self.LMST  = 0


    def image_process( self ):


        if self.filename.split('_N')[0][-3:]=='RZS': 
            self.ftau   = np.float32( self.label['DERIVED_IMAGE_PARMS']['RAD_ZENITH_SCALING_FACTOR'] )
            self.image *= self.ftau

        # if the image has one color band, either demosaic or stack the image to make it a color image
        if len(self.image.shape)==2:
            if 'MV0' in self.IMG_path:
                self.image = np.stack( [self.image,self.image,self.image], axis=-1)
            else:
                # self.image = colour_demosaicing.demosaicing_CFA_Bayer_bilinear  ( self.image, 'RGGB' )
                self.image = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004( self.image, 'RGGB' )
                # self.image = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007 ( im_.image, 'RGGB' )


        d  = 57.296
        try:
            self.mu = np.sin( self.label['SITE_DERIVED_GEOMETRY_PARMS']['SOLAR_ELEVATION'][0]/d )
        except:
            self.mu = 1.0

        self.find_tau()

        self.tau_ref  = 0.3
        self.ftau     = self.mu * np.exp( - ( self.tau - self.tau_ref ) / 6 / self.mu )
        self.ftau_min = 0.2
        if self.ftau  < self.ftau_min: self.ftau = self.ftau_min


        self.im = self.image.copy()
        self.down_sample = self.filename.split('_')[-1][3]

        self.pad_left,self.pad_right,self.pad_top,self.pad_bottom = [0,0,0,0]


        '''
        future work: move these photometric adjustments to a separate function, photo_adjust(...)
        '''
        # Mars2020 Mastcam-Z color processing
        if self.filename[0] == 'Z':

            
            # pad Mastcam-Z images for th non-standard sizes

            self.pad_left,self.pad_right,self.pad_top,self.pad_bottom = [0,0,0,0]

            self.down_sample == '0'
            self.full_height, self.full_width = [ 1200, 1648 ]

            if self.pad_im:
                if self.im.shape != ( self.full_height, self.full_width, 3):                    

                    self.pad_left   =                    self.label['MINI_HEADER']['FIRST_LINE_SAMPLE'] - 1
                    self.pad_right  = self.full_width  - self.label['MINI_HEADER']['LINE_SAMPLES']      - self.label['MINI_HEADER']['FIRST_LINE_SAMPLE']  + 1
                    self.pad_top    =                    self.label['MINI_HEADER']['FIRST_LINE']        - 1
                    self.pad_bottom = self.full_height - self.label['MINI_HEADER']['LINES']             - self.label['MINI_HEADER']['FIRST_LINE']         + 1

                    if self.pad_top!=0 or self.pad_bottom!=0 or self.pad_left!=0 or self.pad_right!=0:
                        
                        print( 'resizing image size {} by padding = [ left, right, top, bottom ] = [ {}, {}, {}, {} ]'.format( \
                            self.image.shape, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ))
                        
                        self.im = pad_image( self.image, pad = [ self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )

        # Mars2020 SuperCam RMI color processing
        # elif self.filename[0]=='L':

            # im /= flat

            # w = 400
            # high_scale = np.percentile( im[w:-w,w:-w,:], 99.8 )
            # im /= high_scale
            # clip_low = np.percentile( im[w:-w,w:-w,:], .05 )
            # clip_low = 0.3
            # high_cut = np.percentile( im[(w+300):-w,w:-w,:], 99.5 )
            # print( 'scale',high_scale, 'cut', clip_low)


        # Ingenuity Return-to-Earch (RTE) color processing
        elif self.filename[0:3] == 'HSF':
            self.ftau = 1.0

            # im /= np.load( 'C:/Users/cornell/Mastcam-Z/ws/HSF/HSF_flat_v1.npy' )
            # im /= np.percentile( im[400:-10,100:-100,:], 99.9 )*1.0
            # w = 100
            # clip_low = 0.2  #np.percentile( im[w:-w,w:-w,:], .5 )


        # Ingenuity Navcam color processing
        elif self.filename[0:3] == 'HNM':
            self.ftau = 1.0
#             w = 50
#             self.im /= np.percentile( im[w:-w,w:-w,:], 99.95 )*1.0
#             clip_low = np.percentile( im[w:-w,w:-w,:], 0.01 )*1.0


        # Mars2020 SHERLOC WATSON color processing
        elif self.filename[0]=='S':
            self.ftau = 1.0
       

        if self.filename[0] in [ 'F', 'N', 'R']:
            # Monochromatic VCE Navcam images
            # if 'MV0' in self.IMG_path:
            #     self.clip_low = 0.25

            # Pad to the image's standard dimensions [ full_height, full_width, 3 ]
            if self.pad_im:
                if   ( self.down_sample == '0' and self.im.shape!=(3840, 5120, 3) ) or \
                     ( self.down_sample == '1' and self.im.shape!=(1920, 2560, 3) ) or \
                     ( self.down_sample == '2' and self.im.shape!=( 960, 1280, 3) ):
                    if self.down_sample == '0': self.full_height, self.full_width = [ 3840, 5120 ]
                    if self.down_sample == '1': self.full_height, self.full_width = [ 1920, 2560 ]
                    if self.down_sample == '2': self.full_height, self.full_width = [  960, 1280 ]

                    self.pad_left   =                    np.min(self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE_SAMPLE']) - 1
                    self.pad_right  = self.full_width  - np.max(self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE_SAMPLE']) - 1280 + 1
                    self.pad_top    =                    np.min(self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE'])        - 1
                    self.pad_bottom = self.full_height - np.max(self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE'])        - 960  + 1
                    
                    if self.pad_right < 0: self.pad_right = 0
                    if self.pad_bottom < 0: self.pad_bottom = 0
                    
                    
                    
                    if self.pad_top!=0 or self.pad_bottom!=0 or self.pad_left!=0 or self.pad_right!=0:
                        
                        print( 'resizing image size {} by padding = [ left, right, top, bottom ] = [ {}, {}, {}, {} ]'.format( \
                            self.image.shape, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ))
                        
                        self.im = pad_image( self.im, pad = [self.pad_left,self.pad_right,self.pad_top,self.pad_bottom] )

        # make image mask
        self.mask_im = self.mask_image.copy() 

        if ( self.pad_top!=0 or self.pad_bottom!=0 or self.pad_left!=0 or self.pad_right!=0 ) and self.pad_im:

            self.mask_im = pad_image( self.mask_im, pad = [ self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )

            if  self.pad_bottom==0 and self.pad_right==0:
                self.mask_im[ self.pad_top:,                 self.pad_left:                ][ self.image[:,:,1] == 0 ] = 0
            elif self.pad_bottom==0:
                self.mask_im[ self.pad_top:,                 self.pad_left:-self.pad_right ][ self.image[:,:,1] == 0 ] = 0
            elif self.pad_right ==0:
                self.mask_im[ self.pad_top:-self.pad_bottom, self.pad_left:                ][ self.image[:,:,1] == 0 ] = 0
            else:
                self.mask_im[ self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right ][ self.image[:,:,1] == 0 ] = 0
        else:
            self.mask_im[ self.image[:,:,1] ==0 ] = 0

        # Mars2020 Mastcam-Z mask processing
        if self.filename[0] in [ 'Z', 'S']:
            
            self.mask_im[ :4,  :] = 0
            self.mask_im[ -1:, :] = 0
            self.mask_im[ : ,:24] = 0
            self.mask_im[ :,-17:] = 0
            
            if 'IOF_N' in self.filename:
                self.ftau = 1.0
                
            # use pre-saved mask
            # parent_path  = os.path.split( os.getcwd() )[0]
            # TODO: Change back when done
            parent_path = os.getcwd()
            if self.filename[:2] == 'ZL':
                mask_path = os.path.join( parent_path, 'params/ZL.jpg' )
            if self.filename[:2] == 'ZR':
                mask_path = os.path.join( parent_path, 'params/ZL.jpg' )
            else:
                mask_path = os.path.join( parent_path, 'params/S.jpg' )
            mask = cv2.imread( mask_path )
            self.mask_im[ mask[:,:,0] < 100 ] = 0

        # Mars2020 SuperCam RMI mask processing        
        if self.filename[0] == 'L':

            self.mask_im[ self.image==0 ] = 0
            self.mask_im[1800:,:,:] = 0
            self.mask_im = cv2.blur( self.mask_im, (20,20))
            self.mask_im[ self.mask_im<255]=0 
            
            
        if self.filename[:3] == 'HNM':

            parent_path  = os.path.split( os.getcwd() )[0]
            mask_path = os.path.join( parent_path, 'params/HNM.jpg' )
            mask = cv2.imread( mask_path )
            self.mask_im[ mask[:,:,0] < 100 ] = 0
            

        # Mars2020 Ecam mask processing    
        else:             
            self.mask_im[  :1, :] = 0
            self.mask_im[ -1:, :] = 0
            self.mask_im[ :, :2 ] = 0
            self.mask_im[ :,-2: ] = 0
            
            
            # use pre-saved mask
            if self.filename[0] == 'F':
            
                parent_path  = os.path.split( os.getcwd() )[0]
                if self.filename[:2] == 'FL':
                    mask_path = os.path.join( parent_path, 'params/FL{}.jpg'.format(self.down_sample) )
                else:
                    mask_path = os.path.join( parent_path, 'params/FR{}.jpg'.format(self.down_sample) )
                
                mask = cv2.imread( mask_path )
                self.mask_im[ mask[:,:,0] < 100 ] = 0
                
            if 'MV0' in self.filename or 'M_0' in self.filename:
            
                parent_path  = os.path.split( os.getcwd() )[0]
                if self.filename[:2] == 'NL':
                    mask_path = os.path.join( parent_path, 'params/NL2_vce.jpg' )
                else:
                    mask_path = os.path.join( parent_path, 'params/NR2_vce.jpg' )
                
                mask = cv2.imread( mask_path )
                self.mask_im[ mask[:,:,0] < 100 ] = 0       
            
            
        if self.filename[:3] == 'HNM' and 1:
           
            im_max = np.max( self.im )
            for i in range(3):
                self.im[:,:,i] = cv2.equalizeHist( np.uint8( self.im[:,:,i] * 255 / im_max )  ).astype('float') / 255 * im_max / self.scale_red
            self.clip_low = 0
            
            
        # apply color and brightnesss corrections
        self.im[:,:,0] *= self.scale / self.ftau * self.scale_red
        self.im[:,:,1] *= self.scale / self.ftau * 1
        self.im[:,:,2] *= self.scale / self.ftau * self.scale_blue
                
        # apply clipping
        self.im = ( self.im - self.clip_low )/( 1 - self.clip_low )
        self.im = np.clip( self.im, 0, 1 )
        
        # apply gamma corection
        if self.gamma != 1.0: 
            self.im = self.im**( 1/self.gamma ) 

        # rescale image to 8 unsigned bits
        self.im8 = np.clip( 255*self.im, 0, 255 ).astype('uint8')
        
        
        
        
        # print( 'processed image', self.filename)


#     def image_reference_perseverance( self ):

#             '''
#             future work: move exterior camera calculations to separate function
#             '''

#             self.az    = self.label['SITE_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'  ][0]
#             self.el    = self.label['SITE_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_ELEVATION'][0]
#             self.xyz   = np.array( self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'].copy() )
#             self.C     = self.label['GEOMETRIC_CAMERA_MODEL']['MODEL_COMPONENT_1'].copy()
#             self.el   += 90
#             self.rl    = 0
#             if self.filename[:2]=='FL': self.rl = + 10
#             if self.filename[:2]=='FR': self.rl = - 10


# #             try: 
# #                 self.rot       = 57.3*np.float32( self.label['RSM_ARTICULATION_STATE']['ARTICULATION_DEVICE_ANGLE'][0] )
# #                 self.rot_rover = (self.label['ROVER_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0] - self.label['SITE_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0])%360
# #             except: 
#             self.rot       = 57.3*np.float32( self.label['RSM_ARTICULATION_STATE']['ARTICULATION_DEVICE_ANGLE'][0][0] )
#             self.rot_rover = ( self.label['ROVER_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0] - 
#                                self.label['SITE_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0])%360

#             self.q   = self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'].copy()
#             self.q   = [ self.q[1], self.q[2], self.q[3], self.q[0]]
#             self.Rot = R.from_quat( self.q )
#             self.Cr  =  self.Rot.apply( self.C, inverse=0 )

#             self.xyz_rover = self.xyz.copy()


#             self.xyz[0] += self.Cr[0]
#             self.xyz[1] += self.Cr[1]
#             self.xyz[2] += self.Cr[2]

#             self.X =  self.xyz[1]
#             self.Y =  self.xyz[0]
#             self.Z = -self.xyz[2]

#             self.X_offset =  self.xyz_rover[1]
#             self.Y_offset =  self.xyz_rover[0]
#             self.Z_offset = -self.xyz_rover[2]


#             self.x_shift, self.y_shift, self.z_shift = xyz_shift_offsets( self.site, self.drive )

#             if 1 and self.find_offsets_mode==0:
#                 self.X += self.x_shift
#                 self.Y += self.y_shift
#                 self.Z += self.z_shift
#                 self.X_offset += self.x_shift
#                 self.Y_offset += self.y_shift
#                 self.Z_offset += self.z_shift
                
                
#     def image_reference_ingenuity( self ):


#         '''
        
        
#         camera pos in M frame = camera model.MODEL_COMPONENT_1
#         camera pos in G frame = camera pos in M frame * M to G quat + M to G offset
#         camera pos in Site3 frame = camera pos in G frame * G to Site3 quat + G to Site3 offset


#         '''


#         self.az    = 0
#         self.el    = 0
#         self.rl    = 0
#         self.rot   = 0 
#         self.rot_rover = 0
        
               
#         self.C    = self.label['GEOMETRIC_CAMERA_MODEL']['MODEL_COMPONENT_1'].copy()

#         self.q   = self.label['HELI_M_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'].copy()
#         self.q   = [ self.q[1], self.q[2], self.q[3], self.q[0]]
#         self.Rot = R.from_quat( self.q )
#         self.Cr  = self.Rot.apply( self.C, inverse=0 )
        
#         self.Cr += np.array( self.label['HELI_M_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'].copy() )
        
#         self.q   = self.label['HELI_G_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'].copy()
#         self.q   = [ self.q[1], self.q[2], self.q[3], self.q[0]]
#         self.Rot = R.from_quat( self.q )
#         self.Cr  = self.Rot.apply( self.Cr, inverse=0 )
                
#         self.Cr += np.array( self.label['HELI_G_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'].copy() )
        
           
#         self.xyz_veh = self.Cr.copy()
        
#         self.q_NED2ENU = R.from_quat( [1,1,0,0] )

#         self.X, self.Y, self.Z = self.q_NED2ENU.apply( self.Cr, inverse=0 )
        
#         self.X_offset, self.Y_offset, self.Z_offset = self.q_NED2ENU.apply( self.xyz_veh, inverse=0 )
        
        

#         self.A = self.label['GEOMETRIC_CAMERA_MODEL']['MODEL_COMPONENT_2']

#         self.q_A = R.from_quat( self.A + [0] )
#         self.q_A = R.from_quat( [0,0,0,1] )


# #         if self.filename[0:3] == 'HSF':
# #             self.q_A = self.q_A * R.from_euler('zyx', np.array([90, 45, 0]), degrees=True)
# #         if self.filename[0:3] == 'HNM':
# #             self.q_A = self.q_A * R.from_euler('zyx', np.array([0, 0, 0]), degrees=True)


#         self.q   = self.label['HELI_M_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'].copy()
#         self.q_M = R.from_quat( [ self.q[1], self.q[2], self.q[3], self.q[0]] )

#         self.q   = self.label['HELI_G_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'].copy()
#         self.q_G = R.from_quat( [ self.q[1], self.q[2], self.q[3], self.q[0]] )

#         self.q_T = self.q_A * self.q_G * self.q_M
# #         self.q_T = self.q_A * self.q_G * self.q_M * self.q_NED2ENU


#         self.ypr = self.q_T.as_euler('zyx', degrees=True)
    
#         self.az  = ( 180 + self.ypr[0] ) % 360        
#         self.el  = 0
#         self.rl  = 0
        
#         self.rot_rover = self.az

# #         print( self.ypr )

        
    def find_tau( self ):
        
        '''
        return the tau, the opacity of Mars sky at the time of the image
        
        future work:
        * access the M2020 tau record
        * estimate the tue for the specific sol
        '''
        
        self.tau = 0.8
        
#         if self.sol >=700:
#             self.tau = 0.5

    # Cmod
    
    def image_reference( self ):
        
        GEOMETRIC_CAMERA_MODEL = self.label['GEOMETRIC_CAMERA_MODEL']
        self.cmod_from_cahvor( GEOMETRIC_CAMERA_MODEL )       
        if self.filename[0] == 'H':
            self.find_Rt_veh2site_inginuity( )
        else:
            self.find_Rt_veh2site_perseverance( )

        self.R_cam2ned  = R.from_matrix( [[0,-1,0],[1,0,0],[0,0,1]] ) 
        self.R_site2cam = self.R_cam2ned * self.R_veh2cam * self.R_veh2site.inv()
        self.ypr = find_ypr_from_R( self.R_site2cam  )    
        self.yaw, self.pitch, self.roll = self.ypr
        self.az, self.el = find_azel_from_ypr( self.ypr  )
        
    def cmod_from_cahvor( self, GEOMETRIC_CAMERA_MODEL ):
        
                   
        self.C  = np.array(  GEOMETRIC_CAMERA_MODEL['MODEL_COMPONENT_1'], dtype=np.float64 )
        self.A  = np.array(  GEOMETRIC_CAMERA_MODEL['MODEL_COMPONENT_2'], dtype=np.float64 )
        self.H  = np.array(  GEOMETRIC_CAMERA_MODEL['MODEL_COMPONENT_3'], dtype=np.float64 )
        self.V  = np.array(  GEOMETRIC_CAMERA_MODEL['MODEL_COMPONENT_4'], dtype=np.float64 )
        self.O  = np.array(  GEOMETRIC_CAMERA_MODEL['MODEL_COMPONENT_5'], dtype=np.float64 )
        self.R  = np.array(  GEOMETRIC_CAMERA_MODEL['MODEL_COMPONENT_6'], dtype=np.float64 )

        self.hs = norm( np.cross( self.H, self.A ) )
        self.vs = norm( np.cross( self.V, self.A ) )
        self.hc = np.dot( self.H, self.A ) 
        self.vc = np.dot( self.V, self.A ) 

        self.hp = ( self.H - self.hc* self.A ) / self.hs
        self.vp = ( self.V - self.vc* self.A ) / self.vs

        self.theta = np.arcsin( ( - norm( np.cross( self.vp, self.hp ) )
                                  / norm( self.vp )
                                  / norm( self.hp ) ) )
        self.theta_degrees = np.rad2deg( self.theta )
        
        self.K_cam = np.array([
                    [ -self.hs*np.sin(self.theta), self.hs*np.cos(self.theta), self.hc ],
                    [                           0,                    self.vs, self.vc ],
                    [                           0,                          0,       1 ], ])

        self.rot_cam = np.matmul( inv( self.K_cam ), 
                                  np.array( [ self.H, self.V, self.A ] ) )
        
        self.R_cam = R.from_matrix( self.rot_cam )       
        self.R_veh2cam = self.R_cam
        
        self.R_ned2enu = R.from_matrix( [[0,1,0],[1,0,0],[0,0,-1]] )
        
        self.w, self.h = [ self.label['IMAGE']['LINE_SAMPLES'], self.label['IMAGE']['LINES'] ]
        if self.w==1600: self.w = 1648

        self.k1 = self.R[1]
        self.k2 = self.R[2]
        self.k3 = 0

        self.p1 = 0
        self.p2 = 0        
        
        self.cxp = self.hc - self.w/2
        self.cyp = self.vc - self.h/2
        
        self.f  =  self.vs
        self.b1 = -self.hs * np.sin( self.theta ) - self.vs
        self.b2 =  self.hs * np.cos( self.theta )

    def find_Rt_veh2site_inginuity( self ):
        
        self.az       = 0
        self.az_veh   = 0
        self.C = self.label['GEOMETRIC_CAMERA_MODEL']['MODEL_COMPONENT_1'].copy()

        self.q_HELI_M = q_wxyz2xyzw( self.label['HELI_M_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'] )
        self.R_HELI_M = R.from_quat( self.q_HELI_M )
        self.C_HELI_M = self.R_HELI_M.apply( self.C, inverse=0 ) \
                      + np.array( self.label['HELI_M_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'] )
                        
        self.q_HELI_G = q_wxyz2xyzw( self.label['HELI_G_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'] )
        self.R_HELI_G = R.from_quat( self.q_HELI_G )
        self.C_HELI_G = self.R_HELI_G.apply( self.C_HELI_M, inverse=0 ) \
                      + np.array( self.label['HELI_G_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'] )
                 
        self.xyz      = self.C_HELI_G.copy()
         
        self.xyz_veh  = self.C_HELI_G.copy()  # approximation

        self.X, self.Y, self.Z = xyz_ned2enu( self.xyz )
        self.X_offset, self.Y_offset, self.Z_offset = xyz_ned2enu( self.xyz_veh )        
        
        self.R_veh2site =  self.R_HELI_M * self.R_HELI_G
        self.t_veh2site = -self.xyz_veh
        
    def find_Rt_veh2site_perseverance( self ):
        self.az       = self.label['SITE_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0]
        self.az_veh = ( self.label['ROVER_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0] - 
                           self.label['SITE_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0])%360

        self.q  = q_wxyz2xyzw( self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'] )
        self.R_veh = R.from_quat( self.q )
        self.Cr =  self.R_veh.apply( self.C, inverse=0 )

        self.xyz_veh = np.array( self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'] )
        
        self.xyz = self.Cr + np.array( self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'] )


#         self.X =  self.xyz[1]
#         self.Y =  self.xyz[0]
#         self.Z = -self.xyz[2]

#         self.X_offset =  self.xyz_veh[1]
#         self.Y_offset =  self.xyz_veh[0]
#         self.Z_offset = -self.xyz_veh[2]

        self.X, self.Y, self.Z = xyz_ned2enu( self.xyz )
        self.X_offset, self.Y_offset, self.Z_offset = xyz_ned2enu( self.xyz_veh )        


        if not self.find_offsets_mode:
            
            self.X_shift, self.Y_shift, self.Z_shift = XYZ_shift_offsets( self.site, self.drive )
            self.X        += self.X_shift
            self.Y        += self.Y_shift
            self.Z        += self.Z_shift
            self.X_offset += self.X_shift
            self.Y_offset += self.Y_shift
            self.Z_offset += self.Z_shift
        
        self.R_veh2site = self.R_veh
        self.t_veh2site = self.xyz
        
                
    
def xyz_ned2enu( xyz ):
    return np.array( [ xyz[1], xyz[0], -xyz[2] ] )
        
def q_wxyz2xyzw( q_wxyz ):
     return np.array([ q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]] )
        
# def find_ypr_from_R( R_site2cam ):    
#     R_cam2ned = R.from_matrix( [[0,-1,0],[1,0,0],[0,0,1]] )    
#     angles = ( R_site2cam * R_cam2ned ).as_euler( 'zxy',degrees=1 )
#     return np.array([ ( - angles[0] ) % 360, - angles[1], angles[2] ])

def find_ypr_from_R( R_ ):  
    return -(R_).as_euler('zyx',degrees=1)

def find_R_from_ypr( ypr ):  
    return R.from_euler( 'zyx', -ypr, degrees=1 )

def find_azel_from_ypr( ypr ):   
    return np.array([ ypr[0] % 360, ypr[1] - 90 ])
    
# def find_R_cam_from_ypr_veh( ypr, R_veh ):    
#     R_cam2ned = R.from_matrix( [[0,-1,0],[1,0,0],[0,0,1]] ) 
#     angles = [ -ypr[0], -ypr[1], ypr[2] ]
#     return R.from_euler( 'zxy', angles, degrees=1 ) * R_cam2ned.inv() * R_veh

# def find_R_cam_from_ypr( ypr ):    
#     R_cam2ned = R.from_matrix( [[0,-1,0],[1,0,0],[0,0,1]] ) 
#     angles = [ -ypr[0], -ypr[1], ypr[2] ]
#     return R.from_euler( 'zxy', angles, degrees=1 ) * R_cam2ned.inv() 


def pad_image( im, pad = [0,0,0,0] ):
    
    if len( im.shape ) == 3:
        im     = np.hstack( [ np.zeros( (im.shape[0],   pad[0] , 3)), im, np.zeros( ( im.shape[0],  pad[1], 3)), ] )
        im     = np.vstack( [ np.zeros( (pad[2]  , im.shape[1],  3)), im, np.zeros( ( pad[3],  im.shape[1], 3)), ] )
    else:
        im     = np.hstack( [ np.zeros( (im.shape[0],   pad[0]    )), im, np.zeros( ( im.shape[0],  pad[1]   )), ] )
        im     = np.vstack( [ np.zeros( (pad[2]  , im.shape[1],   )), im, np.zeros( ( pad[3],  im.shape[1]   )), ] )
    return im

                    

def make_save_path( IMG_path, directory_output, fullpath = True, file_extension = '.png' ):
    
    '''
    make_save_path sorts the images into an output directory organized by camera type and each 100 sols of the mission
    '''
    
    filename = os.path.basename( IMG_path )
    sol      = int( filename[4:8] )
    camera   = filename[0]
    mission  = 'Mars2020' # mission name is hardcoded for now
    
    if camera in ['F','N','R']:
        camera_type = 'eng'
    elif camera in ['H']:
        camera_type = 'heli'
    elif camera in ['Z','L','S']:
        camera_type = 'sci'

    sol_floor_100 = int(np.floor( sol/100 ) * 100)
    sol_range_100 = str(sol_floor_100).zfill(4) + '-' + str(sol_floor_100).zfill(4)[:2] + '99'

    save_path = directory_output + '/sols_' + sol_range_100 + '_' + camera_type 
    
    if not os.path.exists(save_path):
        # Create a new directory because it does not exist
        os.makedirs(save_path)
        print("The new directory is created: ", save_path )

    if fullpath:
        return save_path + '/' + filename.split('.')[0] + file_extension
    else:
        return save_path 



def plot_image_locations( IMG_paths, im_xyzs, rover_xyzs, rover_rots, im_azs, im_els ):
    
    '''
    plot_image_locations displays the Northing vs Easting locations of each image and rover position
    
    future work: replace the input arrays with a single pandas dataframe
    '''
    
    
    plt.figure( figsize=[12,8])
    
    scale = np.round( np.std( np.array(rover_xyzs), axis=0 ).max()/4+1 )

    for i in range(len(im_xyzs)):
        
        filename = os.path.basename( IMG_paths[i] )
        
        marker = '*k'
        if filename[:2] in ['FL','RL']: marker = 'ob'
        if filename[:2] in ['FR','RR']: marker = 'or'
        if filename[:2] ==  'NL':       marker = 'sb'
        if filename[:2] ==  'NR':       marker = 'sr'
        if filename[:2] ==  'ZL':       marker = '^b'
        if filename[:2] ==  'ZR':       marker = '^r'

            
        if 'MV' not in IMG_paths[i]:
            
            plt.plot( rover_xyzs[i][0], rover_xyzs[i][1], color='k',    marker=(4, 0, 45  + rover_rots[i]), ms=30, )
            plt.plot( rover_xyzs[i][0], rover_xyzs[i][1], color='gray', marker=(3, 0, 120 + rover_rots[i]), ms=20, )
        
              
            
            plt.text( rover_xyzs[i][0]+scale/4, rover_xyzs[i][1]+scale/4, 'Sol '+ os.path.basename( IMG_paths[i] )[4:8], \
                      bbox=dict(facecolor='w', alpha=0.5, edgecolor='w'), size='large' )

            if i>1 and os.path.basename( IMG_paths[i] )[:2]=='NLF_':
                plt.plot( [rover_xyzs[i][0], rover_xyzs[i][1] ], [rover_xyzs[i][0], rover_xyzs[i][1] ], '--', color='gray' )
            
            cos_az = np.cos(im_azs[i]/57.3)
            sin_az = np.sin(im_azs[i]/57.3)
            cos_el = np.cos(im_els[i]/57.3)


            if os.path.basename( IMG_paths[i] )[0]=='Z':
                plt.arrow( im_xyzs[i][0], im_xyzs[i][1], scale*cos_el*sin_az, scale*cos_el*cos_az,
                           color=marker[1], lw = int(scale/32), linestyle='dashed' )
            else:
                plt.arrow( im_xyzs[i][0], im_xyzs[i][1], scale*cos_el*sin_az, scale*cos_el*cos_az,
                           color=marker[1], lw = int(scale/32) )
                
        plt.plot( im_xyzs[i][0], im_xyzs[i][1], marker )
        

    plt.axis('equal')
    plt.xlim( [ np.round(plt.gca().get_xlim()[0])-3, np.round(plt.gca().get_xlim()[1])+3 ] )
    plt.ylim( [ np.round(plt.gca().get_ylim()[0])-3, np.round(plt.gca().get_ylim()[1])+3 ] )

    plt.xlabel( 'Easting Site Frame')
    plt.ylabel( 'Northing Site Frame')
#     plt.tight_layout()
#     plt.savefig( path + '/positions'+suf+'.jpg', dpi=300  )




def XYZ_shift_offsets( site, drive ):
    
    '''
    XYZ_shift_offsets finds most accurate Site-Nav offset for each site index and drive
    
    '''

    # print( site, drive )

    # parent_path  = os.path.split( os.getcwd() )[0]
    parent_path = os.getcwd()
    waypoint_shift_path = os.path.join( parent_path, 'params/Mars2020_waypoint_shifts.csv' )

    shift_params = np.loadtxt( waypoint_shift_path, delimiter=',', skiprows=1 )

    site_shifts  = shift_params[ np.where( shift_params[:,1]==site)[0] ]
    site_drives  = site_shifts[:,2]

    if drive in site_drives:
        drive_site_shift = site_shifts[ np.where( site_shifts[:,2]==drive)[0] ][0,:]

    elif drive > site_drives.min() and drive < site_drives.max():
        drive_site_shift = interp1d( site_shifts[:,2], site_shifts, axis=0)(drive)

    elif drive >= site_drives.max():
        drive_site_shift = site_shifts[-1,:]

    else:
        drive_site_shift = np.zeros(12)

    # print( drive_site_shift )
    X_shift, Y_shift, Z_shift = drive_site_shift[9:]

    # X_shift, Y_shift, Z_shift = [ 0,0,0 ]
 
    return X_shift, Y_shift, Z_shift


def remove_duplicate_IMGs( IMG_paths ):
    
    names = [ IMG_paths[i][:-5] for i in range(len(IMG_paths)) ]
    duplicates = list( set(  [ names[i] for i, x in enumerate(names) if i != names.index(x)] ))
    print( len( duplicates ))
    for i in range( len( duplicates) ):
        all_i_paths = sorted( glob.glob( duplicates[i] +'*.IMG'))[::-1]
        duplicates_i_paths = all_i_paths[1:]
        print( 'keeping  ', os.path.basename( all_i_paths[0] ) )
        for j in range(len( duplicates_i_paths )):
            print( 'removing ', os.path.basename( duplicates_i_paths[j] ) )
            os.remove( duplicates_i_paths[j] )
            
            
def image_list_process( IMG_paths, directory_output, suf, find_offsets_mode = 0 ):
    
    
    # File parameters    

    '''
    future work: save these calibration prameters as a text files, which we load for each camera
    '''
    
    file_extension = ''
    
    # save images and thereby overwrite existing images
    save_im    = 1

    # add an alpha channel to the output images
    save_mask  = 1

    # add transparrent pixels to restore the image's full, standard size
    pad_im     = 1
    pad_im_z   = 1

    # turn on when finding the waypoint offsets
#     find_offsets_mode = 1

    # set the color values
    gamma      = 2.2      # gamma value
    gamma      = 2        # gamma value

    # fraction of the dynamic range to clip off the lower values of image 
    clip_low_z = 0.02  # for the Mastcam-Z cameras
    clip_low   = 0.05  # for everything else


    # scale all the scale parameters below bsy the same number
    scale_scale = 18

    # color balance parameters for the Mars 2020 science cameras
    scale_z,  scale_red_z,  scale_blue_z  = [ 1.0*scale_scale, 0.7 , 1.5  ] # Mastcam-Z 
    scale_l,  scale_red_l,  scale_blue_l  = [ 1.0*scale_scale, 0.75, 1.40 ] # SuperCam RMI
    scale_s,  scale_red_s,  scale_blue_s  = [ 1.0*scale_scale, 0.85, 1.40 ] # SHERLOC WATSON 

    # color balance parameters for the Mars 2020 engineering cameras
    scale_n,  scale_red_n,  scale_blue_n  = [ 1.0*scale_scale, 0.75, 1.2  ] # Navcam
    scale_v,  scale_red_v,  scale_blue_v  = [ 1.3*scale_scale, 1.10, 0.93 ] # Grayscale VCE Navcam
    scale_f,  scale_red_f,  scale_blue_f  = [ 1.1*scale_scale, 0.78, 1.25 ] # Front Hazcam
    scale_r,  scale_red_r,  scale_blue_r  = [ 1.1*scale_scale, 0.78, 1.25 ] # Rear Hazcam
    scale_hr, scale_red_hr, scale_blue_hr = [ 1.0*scale_scale, 0.75, 1.43 ] # Inginuity RTE
    scale_hn, scale_red_hn, scale_blue_hn = [ 1.0*scale_scale, 1.08 , 0.92 ] # Inginuity Navcam
    
    pos_lines    = []
    error_lines  = []
    veh_XYZs     = []
    im_XYZs      = []
    veh_azs     = []
    im_azs       = []
    im_els       = []
    sols         = []
    rmcs         = []
    ims          = []
    im_save_path = ''

    print( len(IMG_paths), 'images\n')


    for i in range(len(IMG_paths))[::][:]:

        ####################################################
        ################# *** debugging *** ################
        ####################################################
#         if 1:
        try:    # catch all the images that fail to process

            # open image
            im = image( IMG_paths[i] )
            print( i, im.filename )

            # Set color processing parameters
            im.scale       = scale_scale
            im.scale_red   = 1
            im.scale_blue  = 1
            im.clip_low    = clip_low
            im.gamma       = gamma
            im.pad_im      = pad_im
            im.save_im     = save_im
            im.save_mask   = save_mask
            im.find_offsets_mode = find_offsets_mode

            # Mars 2020 Mastcam-Z
            if im.cam[0] == 'Z':
                im.scale       = scale_z
                im.scale_red   = scale_red_z
                im.scale_blue  = scale_blue_z
                im.clip_low    = clip_low_z
                im.pad_im      = pad_im_z

    #             if 'IOF_N' in im.IMG_path:
    #                 im.scale       = scale_n*1.4
    #                 im.scale_red   = 0.65
    #                 im.scale_blue  = 1.3

            # Mars 2020 SHERLOC WATSON
            if im.cam[0] == 'S':
                im.scale       = scale_s
                im.scale_red   = scale_red_s
                im.scale_blue  = scale_blue_s
                im.clip_low    = 0.0

            # Mars 2020 SuperCam RMI
            if im.cam[0] == 'L':
                im.scale       = scale_l
                im.scale_red   = scale_red_l
                im.scale_blue  = scale_blue_l

            # Mars 2020 Navcam
            if im.cam[0] == 'N':
                im.scale       = scale_n
                im.scale_red   = scale_red_n
                im.scale_blue  = scale_blue_n

                
                
            # Mars 2020 Navcam VCE images
            if 'MV0' in im.filename or 'M_0' in im.filename:
                im.scale       = scale_v
                im.scale_red   = scale_red_v
                im.scale_blue  = scale_blue_v
                im.clip_low    = 0.1


            # Mars 2020 Front Hazcam
            if im.cam[0] == 'F':
                im.scale       = scale_f
                im.scale_red   = scale_red_f
                im.scale_blue  = scale_blue_f
                im.clip_low    = clip_low/2

            # Mars 2020 Rear Hazcam
            if im.cam[0] == 'R':
                im.scale       = scale_r
                im.scale_red   = scale_red_r
                im.scale_blue  = scale_blue_r
                im.clip_low    = clip_low/2

            # Heli Ingenuity RTE 
            if im.filename[0:3] == 'HSF':
                im.scale       = scale_hr
                im.scale_red   = scale_red_hr
                im.scale_blue  = scale_blue_hr

            # Heli Ingenuity Navcam  
            if im.filename[0:3] == 'HNM':
                im.scale       = scale_hn
                im.scale_red   = scale_red_hn
                im.scale_blue  = scale_blue_hn
                im.clip_low    = 0.4
                

            # create save directory
            im.save_path_full = make_save_path( im.IMG_path, directory_output, fullpath=True, file_extension = '.png'  ) 
            im.save_path      = make_save_path( im.IMG_path, directory_output, fullpath=False ) 
            im.save_name      = im.save_path_full.split('/')[-1]
            csv_save_path     = im.save_path_full

            # process and save image
            if im.save_im:

                im.image_process( )

                if im.save_mask:
                    im.im8a = cv2.cvtColor( im.im8, cv2.COLOR_BGR2RGBA )
                    im.im8a[:,:,3] = im.mask_im
                    cv2.imwrite( im.save_path_full, im.im8a )                
                else:
                    cv2.imwrite( im.save_path_full, im.im8[:,:,::-1] )

            # find image position and rotation parameters
            im.image_reference( )


            # save reference data for plotting        
            '''
            future work: replace these lists with pandas dataframes
            '''
            im_XYZs   .append( [ im.X, im.Y, im.Z ] )
            veh_XYZs.append( [ im.X_offset, im.Y_offset, im.Z_offset ] )
            veh_azs.append( im.az_veh )
            im_azs    .append( im.az )
            im_els    .append( im.el )
            rmcs      .append( im.label['ROVER_MOTION_COUNTER'])
            sols      .append( int(im.label['LOCAL_TRUE_SOLAR_TIME_SOL']) )


            # create a line for the reference file
            # Label	 X/East	Y/North	Z/Altitude	Yaw	Pitch	Roll
            pos_line =  im.save_name+'\t'\
                         +str( np.round( im.X,4))+'\t'\
                         +str( np.round( im.Y,4))+'\t'\
                         +str( np.round( im.Z,4))+'\t'\
                         +str( np.round( im.yaw,3))+'\t'\
                         +str( np.round( im.pitch,3))+'\t'\
                         +str( np.round( im.roll,3))+'\n'


            pos_lines.append( pos_line )

            try:
                print( 'sol {} site {} drive {}  zenith angle {:0.0f} scale {:0.2f}'.
                            format( im.sol, im.site, im.drive, im.el*57.3, im.ftau ) )
            except:
                print( 'sol {} site {} drive {}'.
                            format( im.sol, im.site, im.drive, ) )
            print( '', i, pos_line[:], end="\n\n")

        ####################################################
        ################# *** debugging *** ################
        ####################################################
        except:
            print( os.path.basename( IMG_paths[i]), 'failed to process! \n' )
            error_lines.append( os.path.basename( IMG_paths[i]) +'\n' )


    current_time = time.strftime("%Y%m%d-%H%M%S")


    #save failed images list as TXT
    if len(error_lines) > 0:
        csv_save_path = os.path.dirname( csv_save_path)+'/failed_'+suf+'_'+current_time+'.txt'
        with open(csv_save_path,'w') as file:
            for error_line in error_lines:
                file.write(error_line)
    print( 'saved', csv_save_path )

    #save image positions as CSV file
    csv_save_path = os.path.dirname( csv_save_path)+'/positions_'+suf+'_'+current_time+ '.txt'
    with open(csv_save_path,'w') as file:
        for pos_line in pos_lines:
            file.write(pos_line)
    print( 'saved', csv_save_path )

    len( pos_lines )
    
    plot_image_locations( IMG_paths, im_XYZs, veh_XYZs, veh_azs, im_azs, im_els )
    
    if find_offsets_mode:
        sites  = [ rmcs[i][0] for i in range(len(rmcs))[::-1] ]
        drives = [ rmcs[i][1] for i in range(len(rmcs))[::-1] ]
        Xs     = [ veh_XYZs[i][0] for i in range(len(veh_XYZs))[::-1] ]
        Ys     = [ veh_XYZs[i][1] for i in range(len(veh_XYZs))[::-1] ]
        Zs     = [ veh_XYZs[i][2] for i in range(len(veh_XYZs))[::-1] ]

        table = np.stack( [sols[::-1], sites, drives, Xs, Ys, Zs], axis=1)
        np.round( table, 4 )

        np.savetxt( directory_output+"/offsets_"+suf+".csv", table, delimiter="\t")


# add transparrent pixels to restore the image's full, standard size
pad_im = 1
pad_im_z = 1

# turn on when finding the waypoint offsets
#     find_offsets_mode = 1

# set the color values
gamma = 2.2  # gamma value
gamma = 2  # gamma value

# fraction of the dynamic range to clip off the lower values of image
clip_low_z = 0.02  # for the Mastcam-Z cameras
clip_low = 0.05  # for everything else

# scale all the scale parameters below bsy the same number
scale_scale = 18

# color balance parameters for the Mars 2020 science cameras
scale_z, scale_red_z, scale_blue_z = [1.0 * scale_scale, 0.7, 1.5]  # Mastcam-Z
scale_l, scale_red_l, scale_blue_l = [1.0 * scale_scale, 0.75, 1.40]  # SuperCam RMI
scale_s, scale_red_s, scale_blue_s = [1.0 * scale_scale, 0.85, 1.40]  # SHERLOC WATSON

# color balance parameters for the Mars 2020 engineering cameras
scale_n, scale_red_n, scale_blue_n = [1.0 * scale_scale, 0.75, 1.2]  # Navcam
scale_v, scale_red_v, scale_blue_v = [1.3 * scale_scale, 1.10, 0.93]  # Grayscale VCE Navcam
scale_f, scale_red_f, scale_blue_f = [1.1 * scale_scale, 0.78, 1.25]  # Front Hazcam
scale_r, scale_red_r, scale_blue_r = [1.1 * scale_scale, 0.78, 1.25]  # Rear Hazcam
scale_hr, scale_red_hr, scale_blue_hr = [1.0 * scale_scale, 0.75, 1.43]  # Inginuity RTE
scale_hn, scale_red_hn, scale_blue_hn = [1.0 * scale_scale, 1.08, 0.92]  # Inginuity Navcam


def helper_image_list_process_pooled(i, img_path, shared_resources, lock, save_dir):
    """
    Helper function to process [im] within image_list_process_pooled
    @param img_path: filepath to the image to read
    @param shared_resources: dictionary of resources to write to for plotting
    @param: write lock. Only when acquired could we write to the resource
    """
    # checks
    assert os.path.isfile(img_path), f"{img_path} is not a valid path"

    # constant values that are neded
    save_im = 1
    find_offsets_mode = 0

    # add an alpha channel to the output images
    save_mask = 1


    try:

        # open image
        im = image(img_path)
        print(f"{i} Processing {im.filename}...")

        # Set color processing parameters
        im.scale = scale_scale
        im.scale_red = 1
        im.scale_blue = 1
        im.clip_low = clip_low
        im.gamma = gamma
        im.pad_im = pad_im
        im.save_im = save_im
        im.save_mask = save_mask
        im.find_offsets_mode = find_offsets_mode

        # Mars 2020 Mastcam-Z
        if im.cam[0] == 'Z':
            im.scale = scale_z
            im.scale_red = scale_red_z
            im.scale_blue = scale_blue_z
            im.clip_low = clip_low_z
            im.pad_im = pad_im_z

        #             if 'IOF_N' in im.IMG_path:
        #                 im.scale       = scale_n*1.4
        #                 im.scale_red   = 0.65
        #                 im.scale_blue  = 1.3

        # Mars 2020 SHERLOC WATSON
        if im.cam[0] == 'S':
            im.scale = scale_s
            im.scale_red = scale_red_s
            im.scale_blue = scale_blue_s
            im.clip_low = 0.0

        # Mars 2020 SuperCam RMI
        if im.cam[0] == 'L':
            im.scale = scale_l
            im.scale_red = scale_red_l
            im.scale_blue = scale_blue_l

        # Mars 2020 Navcam
        if im.cam[0] == 'N':
            im.scale = scale_n
            im.scale_red = scale_red_n
            im.scale_blue = scale_blue_n

        # Mars 2020 Navcam VCE images
        if 'MV0' in im.filename or 'M_0' in im.filename:
            im.scale = scale_v
            im.scale_red = scale_red_v
            im.scale_blue = scale_blue_v
            im.clip_low = 0.1

        # Mars 2020 Front Hazcam
        if im.cam[0] == 'F':
            im.scale = scale_f
            im.scale_red = scale_red_f
            im.scale_blue = scale_blue_f
            im.clip_low = clip_low / 2

        # Mars 2020 Rear Hazcam
        if im.cam[0] == 'R':
            im.scale = scale_r
            im.scale_red = scale_red_r
            im.scale_blue = scale_blue_r
            im.clip_low = clip_low / 2

        # Heli Ingenuity RTE
        if im.filename[0:3] == 'HSF':
            im.scale = scale_hr
            im.scale_red = scale_red_hr
            im.scale_blue = scale_blue_hr

        # Heli Ingenuity Navcam
        if im.filename[0:3] == 'HNM':
            im.scale = scale_hn
            im.scale_red = scale_red_hn
            im.scale_blue = scale_blue_hn
            im.clip_low = 0.4

        # create save directory
        im.save_path_full = make_save_path(im.IMG_path, save_dir, fullpath=True, file_extension='.png')
        im.save_path = make_save_path(im.IMG_path, save_dir, fullpath=False)
        im.save_name = im.save_path_full.split('/')[-1]
        csv_save_path = im.save_path_full

        # process and save image
        if im.save_im:

            im.image_process()

            if im.save_mask:
                im.im8a = cv2.cvtColor(im.im8, cv2.COLOR_BGR2RGBA)
                im.im8a[:, :, 3] = im.mask_im
                cv2.imwrite(im.save_path_full, im.im8a)
            else:
                cv2.imwrite(im.save_path_full, im.im8[:, :, ::-1])

        # find image position and rotation parameters
        im.image_reference()

        # create a line for the reference file
        # Label	 X/East	Y/North	Z/Altitude	Yaw	Pitch	Roll
        pos_line = im.save_name + '\t' \
                   + str(np.round(im.X, 4)) + '\t' \
                   + str(np.round(im.Y, 4)) + '\t' \
                   + str(np.round(im.Z, 4)) + '\t' \
                   + str(np.round(im.yaw, 3)) + '\t' \
                   + str(np.round(im.pitch, 3)) + '\t' \
                   + str(np.round(im.roll, 3)) + '\n'

        # Acquire write lock to save data for plotting
        with lock:
            shared_resources['im_XYZs'].append([im.X, im.Y, im.Z])
            shared_resources['veh_XYZs'].append([im.X_offset, im.Y_offset, im.Z_offset])
            shared_resources['veh_azs'].append(im.az_veh)
            shared_resources['im_azs'].append(im.az)
            shared_resources['im_els'].append(im.el)
            shared_resources['rmcs'].append(im.label['ROVER_MOTION_COUNTER'])
            shared_resources['sols'].append(int(im.label['LOCAL_TRUE_SOLAR_TIME_SOL']))

            shared_resources['pos_lines'].append(pos_line)

            try:
                print('sol {} site {} drive {}  zenith angle {:0.0f} scale {:0.2f}'.
                      format(im.sol, im.site, im.drive, im.el * 57.3, im.ftau), end="\n\n")

            except:
                print('sol {} site {} drive {}'.
                      format(im.sol, im.site, im.drive, ), end="\n\n")

    except:
        print(os.path.basename(im.IMG_path), 'failed to process! \n')


def image_list_process_pooled(IMG_paths, directory_output, suf, find_offsets_mode=1):
    """
    Similar to image_list_process but pools the job to multiple workers

    future work: save these calibration prameters as a text files, which we load for each camera
    """


    print(len(IMG_paths), 'images\n')

    # multi-processing resources
    pool = Pool()
    manager = Manager()
    lock = manager.Lock()       # lock to be passed around between processes
    # shared resources to write to
    # beyond python 3.6, shared containers can be nested
    shared_resources = manager.dict({
        'pos_lines': manager.list(),
        'error_lines': manager.list(),
        'veh_XYZs': manager.list(),
        'im_XYZs': manager.list(),
        'veh_azs': manager.list(),
        'im_azs': manager.list(),
        'im_els': manager.list(),
        'sols': manager.list(),
        'rmcs': manager.list()
    })

    # assign tasks to each of worker
    for i,im_path in enumerate(IMG_paths):
        pool.apply(helper_image_list_process_pooled,
                         (i, im_path, shared_resources,
                          lock, directory_output))

    # close the pool
    pool.close()
    pool.join()

    # time stamp to print to console
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # save failed images list as TXT
    if len(shared_resources['error_lines']) > 0:
        csv_save_path = directory_output + '/failed_' + suf + '_' + current_time + '.txt'
        with open(csv_save_path, 'w') as file:
            for error_line in shared_resources['error_lines']:
                file.write(error_line)

        print('saved', csv_save_path)

    # save image positions as CSV file
    csv_save_path = directory_output + '/positions_' + suf + '_' + current_time + '.txt'
    with open(csv_save_path, 'w') as file:
        for pos_line in shared_resources['pos_lines']:
            file.write(pos_line)
        print('saved', csv_save_path)

    plot_image_locations(IMG_paths, shared_resources['im_XYZs'], shared_resources['veh_XYZs'],
                         shared_resources['veh_azs'], shared_resources['im_azs'],
                         shared_resources['im_els'])

    if find_offsets_mode:
        sites = [shared_resources['rmcs'][i][0] for i in range(len(shared_resources['rmcs']))[::-1]]
        drives = [shared_resources['rmcs'][i][1] for i in range(len(shared_resources['rmcs']))[::-1]]
        Xs = [shared_resources['veh_XYZs'][i][0] for i in range(len(shared_resources['veh_XYZs']))[::-1]]
        Ys = [shared_resources['veh_XYZs'][i][1] for i in range(len(shared_resources['veh_XYZs']))[::-1]]
        Zs = [shared_resources['veh_XYZs'][i][2] for i in range(len(shared_resources['veh_XYZs']))[::-1]]

        table = np.stack([shared_resources['sols'][::-1], sites, drives, Xs, Ys, Zs], axis=1)
        np.round(table, 4)

        np.savetxt(directory_output + "/offsets_" + suf + ".csv", table, delimiter="\t")
