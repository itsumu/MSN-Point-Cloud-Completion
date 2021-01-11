import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import trans
import h5py
from operations import furthest_point_sample
warnings.filterwarnings('ignore')

def pc_normalize(pc):
	pmax = np.max(pc, axis=0)
	pmin = np.min(pc, axis=0)
	centroid = (pmax + pmin) / 2.0
	return - centroid

class DataLoader(Dataset):
	def __init__(self, root, npoint=2048, split='train', isrotate=False, category=['02691156']):
		self.npoints = npoint
		self.split = split
		self.isrotate = isrotate
		limit_item = 200000

		if split != 'real':
			in_pts1 = np.zeros(shape=(0, 2048, 3))
			in_pts2 = np.zeros(shape=(0, 2048, 3))
			gt_pts1 = np.zeros(shape=(0, 2048, 3))
			gt_pts2 = np.zeros(shape=(0, 2048, 3))
			gt_pts3 = np.zeros(shape=(0, 2048, 3))
			pts_name = np.zeros(shape=(0))
			for cate in category:
				with h5py.File(os.path.join(root, split+'_'+cate+'.h5'), 'r') as f:
					in_pts1 = np.concatenate((in_pts1, np.array(f['in1'])[:limit_item,:,:3]), axis=0).astype(np.float32)
					in_pts2 = np.concatenate((in_pts2, np.array(f['in2'])[:limit_item,:,:3]), axis=0).astype(np.float32)
					gt_pts1 = np.concatenate((gt_pts1, np.array(f['gt1'])[:limit_item,:,:3]), axis=0).astype(np.float32)
					gt_pts2 = np.concatenate((gt_pts2, np.array(f['gt2'])[:limit_item,:,:3]), axis=0).astype(np.float32)
					gt_pts3 = np.concatenate((gt_pts3, np.array(f['gt12'])[:limit_item,:,:3]), axis=0).astype(np.float32)
					pts_name = np.concatenate((pts_name, np.array(f['name'])[:limit_item]), axis=0)
				print(os.path.join(root, split+'_'+cate+'.h5'), ' LOADED!')
			in_pts1[:,:,[1,2]] = in_pts1[:,:,[2,1]]
			in_pts2[:,:,[1,2]] = in_pts2[:,:,[2,1]]
			gt_pts1[:,:,[1,2]] = gt_pts1[:,:,[2,1]]
			gt_pts2[:,:,[1,2]] = gt_pts2[:,:,[2,1]]
			gt_pts3[:,:,[1,2]] = gt_pts3[:,:,[2,1]]
			self.in_ptss1 = np.array(in_pts1)
			self.in_ptss2 = np.array(in_pts2)
			self.gt_ptss1 = np.array(gt_pts1)
			self.gt_ptss2 = np.array(gt_pts2)
			self.gt_ptss3 = np.array(gt_pts3)
			self.ptss_name = np.array(pts_name)
			print('The size of %s data is %d'%(split, len(self.ptss_name)))
		else:
			in_pts1 = np.zeros(shape=(0, 2048, 6))
			in_pts2 = np.zeros(shape=(0, 2048, 6))
			gt_pts1 = np.zeros(shape=(0, 2048, 6))
			gt_pts2 = np.zeros(shape=(0, 2048, 6))
			gt_pts3 = np.zeros(shape=(0, 2048, 6))
			pts_name = np.zeros(shape=(0))
			with h5py.File(os.path.join(root, category[0]+'.h5'), 'r') as f:
				in_pts1 = np.concatenate((in_pts1, np.array(f['in1'])[:limit_item,:,:]), axis=0).astype(np.float32)
				in_pts2 = np.concatenate((in_pts2, np.array(f['in2'])[:limit_item,:,:]), axis=0).astype(np.float32)
				gt_pts1 = np.concatenate((gt_pts1, np.array(f['gt1'])[:limit_item,:,:]), axis=0).astype(np.float32)
				gt_pts2 = np.concatenate((gt_pts2, np.array(f['gt2'])[:limit_item,:,:]), axis=0).astype(np.float32)
				gt_pts3 = np.concatenate((gt_pts3, np.array(f['gt12'])[:limit_item,:,:]), axis=0).astype(np.float32)
				pts_name = np.array(f['name'])
			print(os.path.join(root, category[0]+'.h5'), ' LOADED!')
			in_pts1[:,:,[1,2]] = in_pts1[:,:,[2,1]]
			in_pts2[:,:,[1,2]] = in_pts2[:,:,[2,1]]
			gt_pts1[:,:,[1,2]] = gt_pts1[:,:,[2,1]]
			gt_pts2[:,:,[1,2]] = gt_pts2[:,:,[2,1]]
			gt_pts3[:,:,[1,2]] = gt_pts3[:,:,[2,1]]
			self.in_ptss1 = np.array(in_pts1)
			self.in_ptss2 = np.array(in_pts2)
			self.gt_ptss1 = np.array(gt_pts1)
			self.gt_ptss2 = np.array(gt_pts2)
			self.gt_ptss3 = np.array(gt_pts3)
			self.ptss_name = np.array(pts_name)
			print('The size of %s data is %d'%(split, len(self.ptss_name)))

	def __len__(self):
		return len(self.ptss_name)

	def __getitem__(self, index):
		in_pts1 = self.in_ptss1[index,:int(self.npoints),:3]
		in_pts2 = self.in_ptss2[index,:int(self.npoints),:3]
		gt_pts1 = self.gt_ptss1[index,:int(self.npoints),:3]
		gt_pts2 = self.gt_ptss2[index,:int(self.npoints),:3]
		gt_pts3 = self.gt_ptss3[index,:int(self.npoints),:3]
		# random noise
		# in_pts1 = in_pts1 + np.random.rand(in_pts1.shape[0], in_pts1.shape[1]) * 0.03
		# in_pts2 = in_pts2 + np.random.rand(in_pts2.shape[0], in_pts2.shape[1]) * 0.03

		if self.isrotate:
			np.random.seed()
			angle1 = np.pi / 2 * np.power(np.random.uniform(-1, 1), 3)
			angle2 = np.pi / 2 * np.power(np.random.uniform(-1, 1), 3)
			# angle1 = np.random.uniform(-np.pi/2, np.pi/2)
			# angle2 = np.random.uniform(-np.pi/2, np.pi/2)

			theta1 = np.random.uniform(0, np.pi * 2)
			phi1 = np.random.uniform(0, np.pi / 2)
			x1 = np.cos(theta1) * np.sin(phi1)
			y1 = np.sin(theta1) * np.sin(phi1)
			z1 = np.cos(phi1)
			axis1 = np.array([x1, y1, z1])
			theta2 = np.random.uniform(0, np.pi * 2)
			phi2 = np.random.uniform(0, np.pi / 2)
			x2 = np.cos(theta2) * np.sin(phi2)
			y2 = np.sin(theta2) * np.sin(phi2)
			z2 = np.cos(phi2)
			axis2 = np.array([x2, y2, z2])
		else:
			# angle1 = 0
			angle2 = 0
			# axis1 = np.array([0.0,0.0,1.0])
			axis2 = np.array([0.0,0.0,1.0])
			angle1 = np.pi / 6
			# angle2 = np.pi / 7
			axis1 = np.array([-1.0,-1.0,1.0])
			# axis2 = np.array([1.0,1.0,1.0])

		trans1 = pc_normalize(in_pts1)
		trans2 = pc_normalize(in_pts2)
		centerpoint1 = - trans1
		centerpoint2 = - trans2
		quater1 = trans.axisangle2quaternion(axis=axis1, angle=angle1)
		quater2 = trans.axisangle2quaternion(axis=axis2, angle=angle2)
		matrix1_t = trans.translation2matrix(trans1)
		matrix2_t = trans.translation2matrix(trans2)
		matrix1_r = trans.quaternion2matrix(quater1)
		matrix2_r = trans.quaternion2matrix(quater2)

		matrix1 = np.matmul(matrix1_r, matrix1_t) # First translate, then rotate
		matrix2 = np.matmul(matrix2_r, matrix2_t) # First translate, then rotate

		in_pts1 = trans.transform_pts(in_pts1, matrix1)
		gt_pts1 = trans.transform_pts(gt_pts1, matrix1)
		gt_pts31 = trans.transform_pts(gt_pts3, matrix1)
		centerpoint2 = trans.transform_pts(centerpoint2, matrix1)

		in_pts2 = trans.transform_pts(in_pts2, matrix2)
		gt_pts2 = trans.transform_pts(gt_pts2, matrix2)
		gt_pts32 = trans.transform_pts(gt_pts3, matrix2)
		centerpoint1 = trans.transform_pts(centerpoint1, matrix2)

		gt_para12_r = trans.quat_multiply(quater2, trans.quaternion_inv(quater1))
		gt_para12_t = centerpoint1[0]
		gt_para21_r = trans.quat_multiply(quater1, trans.quaternion_inv(quater2))
		gt_para21_t = centerpoint2[0]

		gt_para_canonical_1 = trans.quaternion_inv(quater1)
		gt_para_canonical_2 = trans.quaternion_inv(quater2)

		if self.split != 'real':
			# return in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2, matrix1, matrix2
			return in_pts1, in_pts2, gt_pts1, gt_pts2
		else:
			color_in_pts1 = self.in_ptss1[index,:int(self.npoints),3:]
			color_in_pts2 = self.in_ptss2[index,:int(self.npoints),3:]
			color_gt_pts1 = self.gt_ptss1[index,:int(self.npoints),3:]
			color_gt_pts2 = self.gt_ptss2[index,:int(self.npoints),3:]
			color_gt_pts3 = self.gt_ptss3[index,:int(self.npoints),3:]
			return in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2, matrix1, matrix2, color_in_pts1, color_in_pts2, color_gt_pts1, color_gt_pts2, color_gt_pts3

	def get_name(self, index):
		# get corresponding pointcloud names from index
		return self.ptss_name[index].decode('utf-8')

if __name__ == '__main__':
	import torch

	testdata = DataLoader(root='../../data/shapenet_color', npoint=2048, split='val', category=['04530566'], isrotate=False)
	DataLoader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
	i = 0
	T_eye = torch.eye(4)
	T_eye = T_eye.reshape((1, 4, 4))
	T_eye = T_eye.repeat(1, 1, 1)
	T_eye = T_eye.cuda()
	for data in DataLoader:
		in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2, matrix1, matrix2 = data
		in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2 = in_pts1.float().cuda(), in_pts2.float().cuda(), gt_pts1.float().cuda(), gt_pts2.float().cuda(), gt_pts31.float().cuda(), gt_pts32.float().cuda(), gt_para12_r.float().cuda(), gt_para12_t.float().cuda(), gt_para21_r.float().cuda(), gt_para21_t.float().cuda(), gt_para_canonical_1.float().cuda(), gt_para_canonical_2.float().cuda()

		if i <= 1:
			R21 = trans.quaternion2matrix_torch(gt_para21_r)
			T21 = trans.translation2matrix_torch(gt_para21_t)
			M21 = torch.bmm(T21, R21)
			R12 = trans.quaternion2matrix_torch(gt_para12_r)
			T12 = trans.translation2matrix_torch(gt_para12_t)
			M12 = torch.bmm(T12, R12)
			gt_rc_pts21_regi = trans.transform_pts_torch(in_pts2, M21)
			gt_rc_pts12_regi = trans.transform_pts_torch(in_pts1, M12)

			in_rc_pts31_comp = torch.cat((in_pts1, gt_rc_pts21_regi), 1)
			in_rc_pts32_comp = torch.cat((in_pts2, gt_rc_pts12_regi), 1)
			np.savetxt('in_rc_pts31_comp.pts', in_rc_pts31_comp.cpu().numpy()[0])
			np.savetxt('in_rc_pts32_comp.pts', in_rc_pts32_comp.cpu().numpy()[0])

			R1_rc = trans.quaternion2matrix_torch(gt_para_canonical_1)
			R2_rc = trans.quaternion2matrix_torch(gt_para_canonical_2)

			gt_pts31_transed = trans.transform_pts_torch(gt_pts31, R1_rc)
			gt_pts32_transed = trans.transform_pts_torch(gt_pts32, R2_rc)

			R1_rc_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(gt_para_canonical_1))#.clone().detach()
			out_rc_pts31_comp = trans.transform_pts_torch(gt_pts31_transed, R1_rc_inv)
			R2_rc_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(gt_para_canonical_2))#.clone().detach()
			out_rc_pts32_comp = trans.transform_pts_torch(gt_pts32_transed, R2_rc_inv)

			out_rc_pts1_final = torch.cat((out_rc_pts31_comp, in_rc_pts31_comp), 1)
			out_rc_pts2_final = torch.cat((out_rc_pts32_comp, in_rc_pts32_comp), 1)

			np.savetxt('out_rc_pts1_final.pts', out_rc_pts1_final.cpu().numpy()[0])
			np.savetxt('out_rc_pts2_final.pts', out_rc_pts2_final.cpu().numpy()[0])

			# R1 = trans.quaternion2matrix_torch(gt_para_canonical_1)
			# in_rc_pts31_comp = torch.cat((in_pts1, gt_rc_pts21_regi), 1)
			# in_rc_pts31_comp_transed = trans.transform_pts_torch(in_rc_pts31_comp, R1)
			# gt_pts31_transed = trans.transform_pts_torch(gt_pts31, R1)
			# in_rc_pts31_comp = torch.cat((in_pts1, gt_rc_pts21_regi), 1)
			# in_rc_pts32_comp = torch.cat((in_pts2, gt_rc_pts12_regi), 1)
			# np.savetxt('in_rc_pts31_comp.pts', in_rc_pts31_comp.cpu().numpy()[0])
			# np.savetxt('in_rc_pts32_comp.pts', in_rc_pts32_comp.cpu().numpy()[0])

			# np.savetxt('gt_pts31.pts', gt_pts31.cpu().numpy()[0])
			# np.savetxt('gt_pts32.pts', gt_pts32.cpu().numpy()[0])
			# np.savetxt('in_rc_pts31_comp_transed.pts', in_rc_pts31_comp_transed.cpu().numpy()[0])
			# np.savetxt('gt_pts31_transed.pts', gt_pts31_transed.cpu().numpy()[0])

			# Generate figure data
			pts_name = testdata.get_name(i)
			# print(pts_name)
			if pts_name == '7228d43e00af4c1e2746490e2236e9a8_0':
				print('get')
				# np.savetxt('in_pts1.pts', in_pts1.cpu().numpy()[0])
				# np.savetxt('in_pts2.pts', in_pts2.cpu().numpy()[0])
				# np.savetxt('gt_pts1.pts', gt_pts1.cpu().numpy()[0])
				# np.savetxt('gt_pts2.pts', gt_pts2.cpu().numpy()[0])				
				R21 = trans.quaternion2matrix(gt_para21_r[0])
				T21 = trans.translation2matrix(gt_para21_t[0])
				M21 = np.matmul(T21, R21)
				R12 = trans.quaternion2matrix(gt_para12_r[0])
				T12 = trans.translation2matrix(gt_para12_t[0])
				M12 = np.matmul(T12, R12)
				R1c = trans.quaternion2matrix(gt_para_canonical_1[0])
				gt_pts1 = np.array(gt_pts1)[0]
				gt_pts2 = np.array(gt_pts2)[0]
				gt_pts31 = np.array(gt_pts31)[0]
				gt_pts32 = np.array(gt_pts32)[0]
				in_pts1 = np.array(in_pts1)[0]
				in_pts2 = np.array(in_pts2)[0]
				in_pts1_2 = trans.transform_pts(in_pts1, R12)
				in_pts1_c = trans.transform_pts(in_pts1, R1c)
				gt_pts1_c = trans.transform_pts(gt_pts1, R1c)
				out_pts1_c =  np.concatenate((in_pts1_c, gt_pts1_c), 0)
				gt_pts1 = gt_pts1 + np.random.rand(gt_pts1.shape[0], gt_pts1.shape[1]) * 0.01
				gt_pts2 = gt_pts2 + np.random.rand(gt_pts2.shape[0], gt_pts2.shape[1]) * 0.01
				out_cr_pts1_mid = np.concatenate((in_pts1, gt_pts1), 0)
				out_cr_pts2_mid = np.concatenate((in_pts2, gt_pts2), 0)
				gt_cr_pts21_regi = trans.transform_pts(out_cr_pts2_mid, M21)
				gt_cr_pts12_regi = trans.transform_pts(out_cr_pts1_mid, M12)
				out_cr_pts1_final = np.concatenate((out_cr_pts1_mid, gt_cr_pts21_regi), 0)[::2,:]
				out_cr_pts2_final = np.concatenate((out_cr_pts2_mid, gt_cr_pts12_regi), 0)[::2,:]

				gt_rc_pts21_regi = trans.transform_pts(in_pts2, M21)
				gt_rc_pts12_regi = trans.transform_pts(in_pts1, M12)
				out_rc_pts31_mid = np.concatenate((in_pts1, gt_rc_pts21_regi), 0)
				out_rc_pts32_mid = np.concatenate((in_pts2, gt_rc_pts12_regi), 0)
				gt_pts31 = gt_pts31 + np.random.rand(gt_pts31.shape[0], gt_pts31.shape[1]) * 0.01
				gt_pts32 = gt_pts32 + np.random.rand(gt_pts32.shape[0], gt_pts32.shape[1]) * 0.01
				out_rc_pts1_final = np.concatenate((out_rc_pts31_mid[::2,:], gt_pts31), 0)
				out_rc_pts2_final = np.concatenate((out_rc_pts32_mid[::2,:], gt_pts32), 0)
				with h5py.File('overall.h5', 'w') as f:
					# f.create_dataset(name="pts_name", data=np.array(pts_name,'S'), compression="gzip")
					f.create_dataset(name="in_pts1", data=np.array(in_pts1), compression="gzip")
					f.create_dataset(name="in_pts2", data=np.array(in_pts2), compression="gzip")
					f.create_dataset(name="out_cr_pts1_mid", data=np.array(out_cr_pts1_mid), compression="gzip")
					f.create_dataset(name="out_cr_pts2_mid", data=np.array(out_cr_pts2_mid), compression="gzip")
					f.create_dataset(name="out_cr_pts1_final", data=np.array(out_cr_pts1_final), compression="gzip")
					f.create_dataset(name="out_cr_pts2_final", data=np.array(out_cr_pts2_final), compression="gzip")
					f.create_dataset(name="out_rc_pts31_mid", data=np.array(out_rc_pts31_mid), compression="gzip")
					f.create_dataset(name="out_rc_pts32_mid", data=np.array(out_rc_pts32_mid), compression="gzip")
					f.create_dataset(name="out_rc_pts1_final", data=np.array(out_rc_pts1_final), compression="gzip")
					f.create_dataset(name="out_rc_pts2_final", data=np.array(out_rc_pts2_final), compression="gzip")
					f.create_dataset(name="in_pts1_2", data=np.array(in_pts1_2), compression="gzip")
					f.create_dataset(name="in_pts1_c", data=np.array(in_pts1_c), compression="gzip")
					f.create_dataset(name="out_pts1_c", data=np.array(out_pts1_c), compression="gzip")
		i += 1
