#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  10 18:07:54 2022

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""
import numpy as np
import torch
import nibabel as nb
from nibabel.gifti import GiftiImage, GiftiDataArray, GiftiCoordSystem, GiftiMetaData,  GiftiNVPairs
from nibabel.filebasedimages import FileBasedHeader
from sklearn.neighbors import KDTree
import math, multiprocessing, os
import itertools
import scipy.io as sio
abspath = os.path.abspath(os.path.dirname(__file__))
ico_dir = 'icosphere/r100/'  
ddr_files_dir = 'DDR_files/' 

ver_ico_dic = {12:0,42:1,162:2,642:3,2562:4,10242:5,40962:6,163842:7,655362:8}


def LossFuns(x, y):
    
    loss_mse = torch.mean(torch.pow((x - y), 2))
    loss_cc  = 1 - ((x - x.mean()) * (y - y.mean())).mean() / x.std() / y.std()
    
    return loss_mse, loss_cc 


def grad_loss(warps, hex_org, device, reg_fun='l1'):
    weights = torch.ones((7, 1), dtype=torch.float32, device = device)
    weights[6] = -6 

    warps_dx = torch.abs(torch.mm(warps[:,[0]][hex_org].view(-1, 7), weights))    
    warps_dy = torch.abs(torch.mm(warps[:,[1]][hex_org].view(-1, 7), weights))    
    warps_dz = torch.abs(torch.mm(warps[:,[2]][hex_org].view(-1, 7), weights))

    if reg_fun == 'l2':
        warps_dx = warps_dx*warps_dx
        warps_dy = warps_dy*warps_dy
        warps_dz = warps_dz*warps_dz

    warps_gradient = (torch.mean(warps_dx) + torch.mean(warps_dy) + torch.mean(warps_dz))/3.0

    return warps_gradient  


def save_gifti(icosphere_array, subject_id, file_to_save): 
       
    ico_level= ver_ico_dic[icosphere_array.shape[0]]
    
    file_name= file_to_save+str(subject_id)+'.DDR.ico-'+str(ico_level)+'.surf.gii'
        
    ico_faces = nb.load(ico_dir+'ico-'+str(ico_level)+'.surf.gii').darrays[1].data 
        
    data_cor = GiftiCoordSystem(dataspace='NIFTI_XFORM_TALAIRACH',
                                xformspace='NIFTI_XFORM_TALAIRACH',
                                xform=None)
 
    nvpair1 = GiftiNVPairs(name='AnatomicalStructureSecondary', value='Invalid')
    nvpair2 = GiftiNVPairs(name='GeometricType', value='Spherical')
     
    Gifti_meta_data=  GiftiMetaData(nvpair1)
    Gifti_meta_data.data.insert(1, nvpair2)
    
    Gifti_data = GiftiDataArray(data= icosphere_array,
                                intent='NIFTI_INTENT_POINTSET',
                                datatype='NIFTI_TYPE_FLOAT32', 
                                encoding='GIFTI_ENCODING_B64GZ', 
                                endian='little', 
                                coordsys=data_cor, 
                                ordering='RowMajorOrder', 
                                meta=Gifti_meta_data, 
                                ext_fname='', 
                                ext_offset=0)
    
    Gifti_cor_face = GiftiCoordSystem(dataspace='NIFTI_XFORM_UNKNOWN', 
                                      xformspace='NIFTI_XFORM_UNKNOWN', 
                                      xform=None)
    
    Gifti_meta_face =  GiftiMetaData()
    
    Gifti_face = GiftiDataArray(data=ico_faces, 
                                intent='NIFTI_INTENT_TRIANGLE', 
                                datatype='NIFTI_TYPE_INT32', 
                                encoding='GIFTI_ENCODING_B64GZ', 
                                endian='little', 
                                coordsys=Gifti_cor_face, 
                                ordering='RowMajorOrder', 
                                meta=Gifti_meta_face, 
                                ext_fname='', 
                                ext_offset=0)
    
    file_head = FileBasedHeader()
    
    my_ico = GiftiImage(header=file_head,
                        extra=None,
                        file_map=None,
                        meta=None,
                        labeltable=None,
                        darrays=[Gifti_data]+[Gifti_face],
                        version='1.0')
    
    nb.save(my_ico, file_name)
    print("Subject '{}' saved!".format(subject_id))

def save_shape_gifti(icosphere_array, subject_id, file_to_save):
    file_name = file_to_save + str(subject_id) + '.DDR.coarse.ico-6' + '.shape.gii'
    file_head = FileBasedHeader()
    Gifti_data = GiftiDataArray(data=icosphere_array,
                                intent='NIFTI_INTENT_SHAPE',
                                datatype='NIFTI_TYPE_FLOAT32',
                                encoding='GIFTI_ENCODING_B64GZ',
                                endian='little',
                                #coordsys=data_cor,
                                ordering='RowMajorOrder',
                                #meta=Gifti_meta_data,
                                ext_fname='',
                                ext_offset=0)
    my_ico = GiftiImage(header=file_head,
                        extra=None,
                        file_map=None,
                        meta=None,
                        labeltable=None,
                        darrays=[Gifti_data],
                        version='1.0')
    nb.save(my_ico, file_name)

def icosphere_upsampling(num_ver, current_ico, next_ver, hex_i, device):
 
    assert current_ico.shape[1] == 3, "icosphere.shape[1] must equal 3"
    assert next_ver == num_ver*4-6, "next_ver ≠ num_ver*4-6"
    
    next_ico = torch.zeros((next_ver, 3), dtype=torch.float, device=device)
    
    next_ico[:num_ver] = current_ico
    next_ico[num_ver:] = torch.mean(current_ico[hex_i],dim=1)

    return next_ico

def get_ico_center(ico_ver,device):
    r_min = torch.min(ico_ver,dim=0)[0].to(device)
    r_max = torch.max(ico_ver,dim=0)[0].to(device)    
    ico_center = (r_min+r_max)/2.0    
    return ico_center


'''
The lat_lon_img Func and the bilinear_sphere_resample Func 
are both inspired by the source code in:   
https://github.com/zhaofenqiang/Spherical_U-Net
'''    

def lat_lon_img(moving_feat, device):
    
    num_ver = len(moving_feat)
    
    img_idxs = np.load(ddr_files_dir+'img_indices_'+ str(num_ver) +'.npy').astype(np.int64)
    img_weights = np.load(ddr_files_dir+'img_weights_'+ str(num_ver) +'.npy').astype(np.float32)
    
    img_idxs =torch.from_numpy(img_idxs).to(device)
    img_weights = torch.from_numpy(img_weights).to(device)    

    W = int(np.sqrt(len(img_idxs)))
    
    img = torch.sum(((moving_feat[img_idxs.flatten()]).reshape(img_idxs.shape[0], img_idxs.shape[1], moving_feat.shape[1]))*((img_weights.unsqueeze(2)).repeat(1,1,moving_feat.shape[1])),1)
    
    img = img.reshape(W, W, moving_feat.shape[1])
    
    return img
            

def bilinear_sphere_resample(rot_grid, org_img, radius, device):
        
    assert rot_grid.shape[1] == 3, "grid.shape[1] ≠ 3"
    
    rot_grid_r1 = rot_grid/radius
    
    w = org_img.shape[0]

    rot_grid_r1[:,2] = torch.clamp(rot_grid_r1[:,2].clone(), -0.9999999, 0.9999999)
    
    Theta = torch.acos(rot_grid_r1[:,2]/1.0)    
    Phi = torch.zeros_like(Theta)
    
    zero_idxs = (rot_grid_r1[:,0] == 0).nonzero(as_tuple=True)[0]
    rot_grid_r1[zero_idxs, 0] = 1e-15
    
    pos_idxs = (rot_grid_r1[:,0] > 0).nonzero(as_tuple=True)[0]
    Phi[pos_idxs] = torch.atan(rot_grid_r1[pos_idxs, 1]/rot_grid_r1[pos_idxs, 0])
    
    neg_idxs = (rot_grid_r1[:,0] < 0).nonzero(as_tuple=True)[0]
    Phi[neg_idxs] = torch.atan(rot_grid_r1[neg_idxs, 1]/rot_grid_r1[neg_idxs, 0]) + math.pi
     
    Phi = torch.remainder(Phi + 2 * math.pi, 2*math.pi)
    # if(len(pos_idxs) + len(neg_idxs) != len(rot_grid_r1)):
    #     print(len(pos_idxs)," ",len(neg_idxs)," ",len(rot_grid_r1))
    #     print(rot_grid_r1)
    assert len(pos_idxs) + len(neg_idxs) == len(rot_grid_r1)
    
    u = Phi/(2*math.pi/(w-1))
    v = Theta/(math.pi/(w-1))
        
    v = torch.clamp(v, 0.0000001, org_img.shape[1]-1.00000001).to(device)
    u = torch.clamp(u, 0.0000001, org_img.shape[1]-1.1).to(device)
    
    u_floor = torch.floor(u)
    u_ceil = u_floor + 1
    v_floor = torch.floor(v)
    v_ceil = v_floor + 1
    
    img1 = org_img[v_floor.long(), u_floor.long()]
    img2 = org_img[v_floor.long(), u_ceil.long()]
    img3 = org_img[v_ceil.long() , u_floor.long()]     
    img4 = org_img[v_ceil.long() , u_ceil.long()]
    
    Q1 = (u_ceil-u).unsqueeze(1)*img1 + (u-u_floor).unsqueeze(1)*img2    
    Q2 = (u_ceil-u).unsqueeze(1)*img3 + (u-u_floor).unsqueeze(1)*img4    
    Q  = (v_ceil-v).unsqueeze(1)*Q1 + (v-v_floor).unsqueeze(1)*Q2
       
    return Q


def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:, 0:6] = adj_mat_order - 1
    neigh_orders[:, 6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)

    return neigh_orders
def isATriangle(neigh_orders, face):
    """
    neigh_orders: int, N x 7
    face: int, 3 x 1
    """
    neighs = neigh_orders[face[0]]
    if face[1] not in neighs or face[2] not in neighs:
        return False
    neighs = neigh_orders[face[1]]
    if face[2] not in neighs:
        return False
    return True
def singleVertexInterpo_ring(vertex, vertices, tree, neigh_orders, ring_iter=1, ring_threshold=3, threshold=1e-8,
                             debug=False):
    if ring_iter > ring_threshold:
        print("ring_iter > ring_threshold, use neaerest 3 neighbor")
        _, top3_near_vertex_index = tree.query(vertex[np.newaxis, :], k=3)
        return np.squeeze(top3_near_vertex_index)

    _, top1_near_vertex_index = tree.query(vertex[np.newaxis, :], k=1)
    ring = []

    if type(neigh_orders) == list:
        ring.append({np.squeeze(top1_near_vertex_index).tolist()})  # 0-ring index
        ring.append(neigh_orders[list(ring[0])[0]])  # 1-ring neighs
        for i in range(ring_iter - 1):
            tmp = set()
            for j in ring[i + 1]:
                tmp = set.union(tmp, neigh_orders[j])
            ring.append(tmp - ring[i] - ring[i + 1])
        candi_vertex = set.union(ring[-1], ring[-2])
    else:
        ring.append(np.squeeze(top1_near_vertex_index))  # 0-ring index
        ring.append(np.setdiff1d(np.unique(neigh_orders[ring[0]]), ring[0]))  # 1-ring neighs
        for i in range(ring_iter - 1):
            tmp = np.setdiff1d(np.unique(neigh_orders[ring[i + 1]].flatten()), ring[i + 1])
            ring.append(np.setdiff1d(tmp, ring[i]))
        candi_vertex = np.append(ring[-1], ring[-2])

    candi_faces = []
    for t in itertools.combinations(candi_vertex, 3):
        tmp = np.asarray(t)  # get the indices of the potential candidate triangles
        if isATriangle(neigh_orders, tmp):
            candi_faces.append(tmp)
    candi_faces = np.asarray(candi_faces)

    orig_vertex_1 = vertices[candi_faces[:, 0]]
    orig_vertex_2 = vertices[candi_faces[:, 1]]
    orig_vertex_3 = vertices[candi_faces[:, 2]]
    edge_12 = orig_vertex_2 - orig_vertex_1  # edge vectors from vertex 1 to 2
    edge_13 = orig_vertex_3 - orig_vertex_1  # edge vectors from vertex 1 to 3
    faces_normal = np.cross(edge_12, edge_13)  # normals of all the faces
    tmp = (np.linalg.norm(faces_normal, axis=1) == 0).nonzero()[0]
    faces_normal[tmp] = orig_vertex_1[tmp]
    faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:, np.newaxis]

    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
    tmp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
    P = tmp[:, np.newaxis] * vertex  # intersection points

    # find the triangle face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole one
    area_BCP = np.linalg.norm(np.cross(orig_vertex_3 - P, orig_vertex_2 - P), axis=1) / 2.0
    area_ACP = np.linalg.norm(np.cross(orig_vertex_3 - P, orig_vertex_1 - P), axis=1) / 2.0
    area_ABP = np.linalg.norm(np.cross(orig_vertex_2 - P, orig_vertex_1 - P), axis=1) / 2.0
    area_ABC = np.linalg.norm(faces_normal, axis=1) / 2.0

    tmp = area_BCP + area_ACP + area_ABP - area_ABC
    index = np.argmin(tmp)

    if tmp[index] > threshold:
        return singleVertexInterpo_ring(vertex, vertices, tree, neigh_orders,
                                        ring_iter=ring_iter + 1, ring_threshold=ring_threshold,
                                        threshold=threshold, debug=False)

    return candi_faces[index]


# deprecated
# def singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=7, threshold=1e-6, k_threshold=15, debug=False):

#     if k > k_threshold:
#         # print("use neaerest neighbor, k=", k)
#         _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
#         top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
#         return top3_near_vertex_index

#     _, top7_near_vertex_index = tree.query(vertex[np.newaxis,:], k=k)
#     candi_faces = []
#     for t in itertools.combinations(np.squeeze(top7_near_vertex_index), 3):
#         tmp = np.asarray(t)  # get the indices of the potential candidate triangles
#         if isATriangle(neigh_orders, tmp):
#              candi_faces.append(tmp)

#     if candi_faces:
#         candi_faces = np.asarray(candi_faces)
#     else:
#         if k > k_threshold-5 and debug==True:
#             print("cannot find candidate faces, top k shoulb be larger, function recursion, current k =", k)
#         return singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=k+5,
#                                      threshold=threshold, k_threshold=k_threshold)

#     orig_vertex_1 = vertices[candi_faces[:,0]]
#     orig_vertex_2 = vertices[candi_faces[:,1]]
#     orig_vertex_3 = vertices[candi_faces[:,2]]
#     edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
#     edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
#     faces_normal = np.cross(edge_12, edge_13)    # normals of all the faces
#     tmp = (np.linalg.norm(faces_normal, axis=1) == 0).nonzero()[0]
#     faces_normal[tmp] = orig_vertex_1[tmp]
#     faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:,np.newaxis]

#     # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
#     tmp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
#     P = tmp[:, np.newaxis] * vertex  # intersection points

#     # find the triangle face that the inersection is in, if the intersection
#     # is in, the area of 3 small triangles is equal to the whole one
#     area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
#     area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABC = np.linalg.norm(faces_normal, axis=1)/2.0

#     tmp = area_BCP + area_ACP + area_ABP - area_ABC
#     index = np.argmin(tmp)

#     if tmp[index] > threshold:
#         if k > 30 and debug==True:
#             print("candidate faces don't contain the correct one, top k shoulb be larger, function recursion, current k =", k)
#         return singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=k+5,
#                                      threshold=threshold, k_threshold=k_threshold)

#     w = np.array([area_BCP[index], area_ACP[index], area_ABP[index]])
#     if w.sum() == 0:
#         _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
#         top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
#         return top3_near_vertex_index
#     else:
#         return candi_faces[index]


def singleVertexInterpo(vertex, vertices, tree, neigh_orders, feat, fast, threshold=1e-8, ring_threshold=3):
    """
    Compute the three indices for sphere interpolation at given position.

    """
    _, top3_near_vertex_index = tree.query(vertex[np.newaxis, :], k=3)
    top3_near_vertex_index = np.squeeze(top3_near_vertex_index)

    if fast:
        return top3_near_vertex_index

    if isATriangle(neigh_orders, top3_near_vertex_index):
        v0 = vertices[top3_near_vertex_index[0]]
        v1 = vertices[top3_near_vertex_index[1]]
        v2 = vertices[top3_near_vertex_index[2]]

        normal = np.cross(v1 - v2, v0 - v2)
        vertex_proj = v0.dot(normal) / vertex.dot(normal) * vertex
        area_BCP = np.linalg.norm(np.cross(v2 - vertex_proj, v1 - vertex_proj)) / 2.0
        area_ACP = np.linalg.norm(np.cross(v2 - vertex_proj, v0 - vertex_proj)) / 2.0
        area_ABP = np.linalg.norm(np.cross(v1 - vertex_proj, v0 - vertex_proj)) / 2.0
        area_ABC = np.linalg.norm(normal) / 2.0

        if area_BCP + area_ACP + area_ABP - area_ABC > threshold:
            # inter_indices = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders,
            #                                       threshold=threshold, k_threshold=k_threshold)
            inter_indices = singleVertexInterpo_ring(vertex, vertices, tree,
                                                     neigh_orders, ring_iter=1,
                                                     ring_threshold=ring_threshold,
                                                     threshold=threshold, debug=False)
        else:
            inter_indices = top3_near_vertex_index

    else:
        # inter_indices = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders,
        #                                       threshold=threshold, k_threshold=k_threshold)
        inter_indices = singleVertexInterpo_ring(vertex, vertices, tree,
                                                 neigh_orders, ring_iter=1,
                                                 ring_threshold=ring_threshold,
                                                 threshold=threshold, debug=False)
    return inter_indices


def multiVertexInterpo(vertexs, vertices, tree, neigh_orders, feat, fast, threshold, ring_threshold):
    inter_indices = np.zeros((vertexs.shape[0], 3), dtype=np.int32)

    for i in range(vertexs.shape[0]):
        inter_indices[i, :] = singleVertexInterpo(vertexs[i, :],
                                                  vertices,
                                                  tree,
                                                  neigh_orders,
                                                  feat,
                                                  fast,
                                                  threshold,
                                                  ring_threshold)
    return inter_indices
def multiVertexInterpo(vertexs, vertices, tree, neigh_orders, feat, fast, threshold, ring_threshold):
    inter_indices = np.zeros((vertexs.shape[0], 3), dtype=np.int32)

    for i in range(vertexs.shape[0]):
        inter_indices[i, :] = singleVertexInterpo(vertexs[i, :],
                                                  vertices,
                                                  tree,
                                                  neigh_orders,
                                                  feat,
                                                  fast,
                                                  threshold,
                                                  ring_threshold)
    return inter_indices

def resampleStdSphereSurf(n_curr, n_next, feat, upsample_neighbors):
    assert len(feat) == n_curr, "feat length not cosistent!"
    assert n_next == n_curr * 4 - 6, "This function can only upsample one level higher" + \
                                     " If you want to upsample with two levels higher, you need to call this function twice."

    feat_inter = np.zeros((n_next, feat.shape[1]))
    feat_inter[0:n_curr, :] = feat
    feat_inter[n_curr:, :] = feat[upsample_neighbors].reshape(n_next - n_curr, 2, feat.shape[1]).mean(1)

    return feat_inter

def resampleSphereSurf(vertices_fix, vertices_inter, feat, faces=None,
                       std=False, upsample_neighbors=None, neigh_orders=None,
                       fast=False, threshold=1e-6, ring_threshold=3):
    """
    resample sphere surface
    Parameters
    ----------
    vertices_fix :  N*3, numpy array,
        the original fixed vertices with features.
    vertices_inter : unknown*3, numpy array,
        points to be interpolated.
    feat :  N*D,
        features to be interpolated.
    faces :  N*4, numpy array, the first column shoud be all 3
        is the original faces directly read using read_vtk,. The default is None.
    std : bool
        standard sphere interpolation, e.g., interpolate 10242 from 2562.. The default is False.
    upsample_neighbors : TYPE, optional
        DESCRIPTION. The default is None.
    neigh_orders : TYPE, optional
        DESCRIPTION. The default is None.
    fast : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    Returns
    -------
    resampled feature.

    """

    assert vertices_fix.shape[0] == feat.shape[0], "vertices.shape[0] is not equal to feat.shape[0]"
    assert vertices_fix.shape[1] == 3, "vertices size not right"

    # vertices_fix = vertices_fix.astype(np.float64)
    # vertices_inter = vertices_inter.astype(np.float64)
    # feat = feat.astype(np.float64)
    vertices_fix = vertices_fix.detach().cpu().numpy().astype(np.float64)
    vertices_inter = vertices_inter.detach().cpu().numpy().astype(np.float64)
    feat = feat.detach().cpu().numpy().astype(np.float64)
    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:, np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:, np.newaxis]  # normalize to 1

    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]

    if std:
        assert upsample_neighbors is not None, " upsample_neighbors is None"
        return resampleStdSphereSurf(len(vertices_fix), len(vertices_inter), feat, upsample_neighbors)

    if not fast:
        if neigh_orders is None:
            if faces is not None:
                assert faces.shape[1] == 4, "faces shape is wrong, should be N*4"
                assert (faces[:, 0] == 3).sum() == faces.shape[0], "the first column of faces should be all 3"
                faces = faces[:, 1:]

                num_vers = vertices_fix.shape[0]
                neigh_unsorted_orders = []
                for i in range(num_vers):
                    neigh_unsorted_orders.append(set())
                for i in range(faces.shape[0]):
                    face = faces[i]
                    neigh_unsorted_orders[face[0]].add(face[1])
                    neigh_unsorted_orders[face[0]].add(face[2])
                    neigh_unsorted_orders[face[1]].add(face[0])
                    neigh_unsorted_orders[face[1]].add(face[2])
                    neigh_unsorted_orders[face[2]].add(face[0])
                    neigh_unsorted_orders[face[2]].add(face[1])

                neigh_orders = neigh_unsorted_orders

                # deprecated, too slow, use above set() method
                # for i in range(faces.shape[0]):
                #     if faces[i,1] not in neigh_orders[faces[i,0]]:
                #         neigh_orders[faces[i,0], np.where(neigh_orders[faces[i,0]] == -1)[0][0]] = faces[i,1]
                #     if faces[i,2] not in neigh_orders[faces[i,0]]:
                #         neigh_orders[faces[i,0], np.where(neigh_orders[faces[i,0]] == -1)[0][0]] = faces[i,2]
                #     if faces[i,0] not in neigh_orders[faces[i,1]]:
                #         neigh_orders[faces[i,1], np.where(neigh_orders[faces[i,1]] == -1)[0][0]] = faces[i,0]
                #     if faces[i,2] not in neigh_orders[faces[i,1]]:
                #         neigh_orders[faces[i,1], np.where(neigh_orders[faces[i,1]] == -1)[0][0]] = faces[i,2]
                #     if faces[i,1] not in neigh_orders[faces[i,2]]:
                #         neigh_orders[faces[i,2], np.where(neigh_orders[faces[i,2]] == -1)[0][0]] = faces[i,1]
                #     if faces[i,0] not in neigh_orders[faces[i,2]]:
                #         neigh_orders[faces[i,2], np.where(neigh_orders[faces[i,2]] == -1)[0][0]] = faces[i,0]

            else:
                neigh_orders = get_neighs_order(
                    abspath + '/neigh_indices/adj_mat_order_' + str(vertices_fix.shape[0]) + '_rotated_0.mat')
                neigh_orders = neigh_orders.reshape(vertices_fix.shape[0], 7)
        else:
            neigh_orders = neigh_orders.reshape(vertices_fix.shape[0], 7)

    inter_indices = np.zeros((vertices_inter.shape[0], 3), dtype=np.int32)
    tree = KDTree(vertices_fix, leaf_size=10)  # build kdtree

    """ Single process, single thread: 163842: 54.5s, 40962: 12.7s, 10242: 3.2s, 2562: 0.8s """
    #    for i in range(vertices_inter.shape[0]):
    #        print(i)
    #        inter_indices[i,:] = singleVertexInterpo(vertices_inter[i,:], vertices_fix, tree, neigh_orders, feat)

    """ multiple processes method: 163842:  s, 40962: 2.8s, 10242: 1.0s, 2562: 0.28s """
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    vertexs_num_per_cpu = math.ceil(vertices_inter.shape[0] / cpus)
    results = []

    for i in range(cpus):
        results.append(pool.apply_async(multiVertexInterpo,
                                        args=(vertices_inter[i * vertexs_num_per_cpu:(i + 1) * vertexs_num_per_cpu, :],
                                              vertices_fix, tree, neigh_orders, feat, fast, threshold, ring_threshold)))

    pool.close()
    pool.join()

    for i in range(cpus):
        inter_indices[i * vertexs_num_per_cpu:(i + 1) * vertexs_num_per_cpu, :] = results[i].get()

    v0 = vertices_fix[inter_indices[:, 0]]
    v1 = vertices_fix[inter_indices[:, 1]]
    v2 = vertices_fix[inter_indices[:, 2]]
    normal = np.cross(v1 - v0, v2 - v0, axis=1)
    vertex_proj = np.sum(v0 * normal, axis=1, keepdims=True) / np.sum(vertices_inter * normal, axis=1,
                                                                      keepdims=True) * vertices_inter

    tmp_index = np.argwhere(np.isnan(vertex_proj))[:, 0]  # in case that normal is [0,0,0]

    area_12P = np.linalg.norm(np.cross(v2 - vertex_proj, v1 - vertex_proj, axis=1), axis=1, keepdims=True) / 2.0
    area_02P = np.linalg.norm(np.cross(v2 - vertex_proj, v0 - vertex_proj, axis=1), axis=1, keepdims=True) / 2.0
    area_01P = np.linalg.norm(np.cross(v1 - vertex_proj, v0 - vertex_proj, axis=1), axis=1, keepdims=True) / 2.0
    inter_weights = np.concatenate(([area_12P, area_02P, area_01P]), axis=1)

    inter_weights[tmp_index] = np.array([1, 0, 0])  # in case that normal is [0,0,0]

    inter_weights = inter_weights / np.sum(inter_weights, axis=1, keepdims=True)

    feat_inter = np.sum(
        np.multiply(feat[inter_indices], np.repeat(inter_weights[:, :, np.newaxis], feat.shape[1], axis=2)), axis=1)

    return feat_inter
