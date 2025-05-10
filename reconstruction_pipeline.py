import os
import numpy as np
import cv2
import open3d as o3d
from feature_extraction import FeatureExtractor
from camera_estimation import CameraEstimator
from point_cloud import PointCloudGenerator
from mesh_generator import MeshGenerator
from texture_mapper import TextureMapper

class ReconstructionPipeline:
    def __init__(self, feature_type='sift', skip_dense=False):
        self.feature_type = feature_type
        self.skip_dense = skip_dense
        self.feature_extractor = FeatureExtractor(feature_type)
        self.camera_estimator = CameraEstimator()
        self.point_cloud_generator = PointCloudGenerator()
        self.mesh_generator = MeshGenerator()
        self.texture_mapper = TextureMapper()
        
    def run(self, input_dir, output_dir):
        # Step 1: Load and preprocess images
        images, image_paths = self._load_images(input_dir)
        print(f"Loaded {len(images)} images")
        
        # Step 2: Extract features and find matches
        features, matches = self.feature_extractor.process_images(images)
        print(f"Extracted features and found matches")
        
        # Step 3: Estimate camera poses
        cameras = self.camera_estimator.estimate_poses(features, matches)
        print(f"Estimated camera poses")
        
        # Step 4: Generate sparse point cloud
        sparse_cloud = self.point_cloud_generator.generate_sparse(features, matches, cameras)
        self._save_point_cloud(sparse_cloud, os.path.join(output_dir, "sparse_cloud.ply"))
        print(f"Generated sparse point cloud")
        
        if not self.skip_dense:
            # Step 5: Generate dense point cloud
            dense_cloud = self.point_cloud_generator.generate_dense(images, cameras)
            self._save_point_cloud(dense_cloud, os.path.join(output_dir, "dense_cloud.ply"))
            print(f"Generated dense point cloud")
            
            # Step 6: Generate mesh
            mesh = self.mesh_generator.create_mesh(dense_cloud)
            self._save_mesh(mesh, os.path.join(output_dir, "mesh.obj"))
            print(f"Generated mesh")
            
            # Step 7: Apply textures
            textured_mesh = self.texture_mapper.apply_textures(mesh, images, cameras, image_paths)
            self._save_mesh(textured_mesh, os.path.join(output_dir, "textured_mesh.obj"))
            print(f"Applied textures")
        
        return True
    
    def _load_images(self, input_dir):
        """Load and preprocess images from the input directory"""
        images = []
        image_paths = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(input_dir, filename)
                img = cv2.imread(filepath)
                
                if img is not None:
                    # Convert to RGB (OpenCV loads as BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    image_paths.append(filepath)
        
        return images, image_paths
    
    def _save_point_cloud(self, point_cloud, filepath):
        """Save point cloud to file"""
        o3d.io.write_point_cloud(filepath, point_cloud)
    
    def _save_mesh(self, mesh, filepath):
        """Save mesh to file"""
        o3d.io.write_triangle_mesh(filepath, mesh)
