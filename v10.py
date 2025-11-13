import cv2
import numpy as np
import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
from itertools import product


class AdvancedImageReconstructor:
    def __init__(self, db_name="image_reconstruction.db"):
        """Initialize the reconstructor with a database connection."""
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

        self.cursor.execute("DROP TABLE IF EXISTS reconstruction_metadata")
        self.cursor.execute("DROP TABLE IF EXISTS reconstructions")

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reconstructions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image1_path TEXT NOT NULL,
                image2_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                method TEXT,
                rotation_detected REAL
            )
        """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reconstruction_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reconstruction_id INTEGER,
                keypoints_found INTEGER,
                matches_found INTEGER,
                confidence REAL,
                FOREIGN KEY (reconstruction_id)
                    REFERENCES reconstructions(id)
            )
        """
        )

        self.conn.commit()

    def load_images(self, image1_path, image2_path):
        """Load two images from file paths."""
        if not os.path.exists(image1_path):
            raise FileNotFoundError(f"Image 1 not found: {image1_path}")
        if not os.path.exists(image2_path):
            raise FileNotFoundError(f"Image 2 not found: {image2_path}")

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")

        return img1, img2

    def rotate_image(self, img, angle):
        """Rotate image by angle (0, 90, 180, 270 degrees)."""
        if angle == 0:
            return img.copy()
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def flip_image(self, img, flip_code):
        """Flip image (0=vertical, 1=horizontal, -1=both)."""
        if flip_code is None:
            return img.copy()
        return cv2.flip(img, flip_code)

    def extract_edge(self, img, edge, width=50):
        """Extract an edge strip from image.
        
        Args:
            img: Input image
            edge: 'top', 'bottom', 'left', 'right'
            width: Width/height of the strip to extract
        """
        h, w = img.shape[:2]
        width = min(width, min(h, w) // 3)
        
        if edge == 'top':
            return img[:width, :]
        elif edge == 'bottom':
            return img[-width:, :]
        elif edge == 'left':
            return img[:, :width]
        elif edge == 'right':
            return img[:, -width:]
        return None

    def compare_edges(self, edge1, edge2, edge1_type, edge2_type):
        """Compare two edge strips and return similarity score.
        
        Compatible edges:
        - top <-> bottom
        - bottom <-> top
        - left <-> right
        - right <-> left
        """
        # Check if edges are compatible
        valid_pairs = [
            ('top', 'bottom'), ('bottom', 'top'),
            ('left', 'right'), ('right', 'left')
        ]
        
        if (edge1_type, edge2_type) not in valid_pairs:
            return -1.0  # Incompatible edges
        
        # Resize edges to match
        if edge1_type in ['top', 'bottom']:
            # Horizontal edges - match widths
            h = min(edge1.shape[0], edge2.shape[0])
            w = min(edge1.shape[1], edge2.shape[1])
            edge1_resized = cv2.resize(edge1, (w, h))
            edge2_resized = cv2.resize(edge2, (w, h))
        else:
            # Vertical edges - match heights
            h = min(edge1.shape[0], edge2.shape[0])
            w = min(edge1.shape[1], edge2.shape[1])
            edge1_resized = cv2.resize(edge1, (w, h))
            edge2_resized = cv2.resize(edge2, (w, h))
        
        # Convert to grayscale for comparison
        if len(edge1_resized.shape) == 3:
            edge1_gray = cv2.cvtColor(edge1_resized, cv2.COLOR_BGR2GRAY)
        else:
            edge1_gray = edge1_resized
            
        if len(edge2_resized.shape) == 3:
            edge2_gray = cv2.cvtColor(edge2_resized, cv2.COLOR_BGR2GRAY)
        else:
            edge2_gray = edge2_resized
        
        # Calculate multiple similarity metrics
        
        # 1. Normalized cross-correlation
        edge1_norm = (edge1_gray - edge1_gray.mean()) / (edge1_gray.std() + 1e-8)
        edge2_norm = (edge2_gray - edge2_gray.mean()) / (edge2_gray.std() + 1e-8)
        correlation = np.mean(edge1_norm * edge2_norm)
        
        # 2. Structural similarity (simplified)
        diff = np.abs(edge1_gray.astype(float) - edge2_gray.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        # 3. Edge-based similarity
        edge1_edges = cv2.Canny(edge1_gray, 50, 150)
        edge2_edges = cv2.Canny(edge2_gray, 50, 150)
        edge_similarity = 1.0 - (
            np.sum(np.abs(edge1_edges - edge2_edges)) / 
            (edge1_edges.size * 255.0)
        )
        
        # Combined score
        score = (correlation * 0.4 + similarity * 0.4 + edge_similarity * 0.2)
        
        return score

    def find_best_configuration(self, img1, img2):
        """Try all possible orientations and find best match."""
        print(f"\n  Testing all possible configurations...")
        
        best_score = -np.inf
        best_config = None
        
        # All possible rotations (0, 90, 180, 270)
        rotations = [0, 90, 180, 270]
        # All possible flips (None, horizontal, vertical, both)
        flips = [None, 1, 0, -1]
        # All possible edges
        edges = ['top', 'bottom', 'left', 'right']
        
        configurations = []
        
        # Try all combinations for img1 and img2
        for rot1, flip1, rot2, flip2 in product(
            rotations, flips, rotations, flips
        ):
            # Apply transformations
            img1_trans = self.rotate_image(img1, rot1)
            if flip1 is not None:
                img1_trans = self.flip_image(img1_trans, flip1)
            
            img2_trans = self.rotate_image(img2, rot2)
            if flip2 is not None:
                img2_trans = self.flip_image(img2_trans, flip2)
            
            # Try all edge combinations
            for edge1, edge2 in product(edges, edges):
                strip1 = self.extract_edge(img1_trans, edge1)
                strip2 = self.extract_edge(img2_trans, edge2)
                
                score = self.compare_edges(strip1, strip2, edge1, edge2)
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'img1_rot': rot1,
                        'img1_flip': flip1,
                        'img2_rot': rot2,
                        'img2_flip': flip2,
                        'edge1': edge1,
                        'edge2': edge2,
                        'score': score,
                        'img1_trans': img1_trans,
                        'img2_trans': img2_trans
                    }
                
                configurations.append(score)
        
        print(f"  Tested {len(configurations)} configurations")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best config: Img1[rot={best_config['img1_rot']}°, "
              f"flip={best_config['img1_flip']}] <-> "
              f"Img2[rot={best_config['img2_rot']}°, "
              f"flip={best_config['img2_flip']}]")
        print(f"  Edge match: {best_config['edge1']} <-> "
              f"{best_config['edge2']}")
        
        return best_config

    def preprocess_image(self, img):
        """Preprocess image for better feature detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def detect_features_robust(self, img1, img2):
        """Enhanced feature detection with SIFT and ORB."""
        gray1 = self.preprocess_image(img1)
        gray2 = self.preprocess_image(img2)

        try:
            # Try SIFT first (better for complex images)
            sift = cv2.SIFT_create(
                nfeatures=5000,
                contrastThreshold=0.03,
                edgeThreshold=10
            )
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            method = "SIFT"
        except (cv2.error, AttributeError):
            # Fallback to ORB
            orb = cv2.ORB_create(nfeatures=20000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            method = "ORB"

        return kp1, des1, kp2, des2, method

    def match_features_robust(self, des1, des2, method):
        """Enhanced feature matching with ratio test."""
        if des1 is None or des2 is None:
            return []
        
        if method == "SIFT":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            ratio = 0.7
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            ratio = 0.75

        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return []

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)

        return good_matches

    def validate_homography(self, H, img1_shape, img2_shape):
        """Validate if homography makes sense for torn image reconstruction."""
        if H is None:
            return False
        
        # Check if homography is too extreme
        # For torn images, we expect mostly translation with minor rotation/scale
        
        # Extract scale, rotation, and translation
        try:
            # Decompose homography (simplified check)
            h1, w1 = img1_shape[:2]
            h2, w2 = img2_shape[:2]
            
            # Check corners transformation
            corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)
            
            # Calculate distances between adjacent corners
            distances = []
            for i in range(4):
                p1 = transformed[i][0]
                p2 = transformed[(i + 1) % 4][0]
                dist = np.linalg.norm(p1 - p2)
                distances.append(dist)
            
            # Original distances
            orig_distances = [w2, h2, w2, h2]
            
            # Check if scaling is reasonable (not more than 2x or less than 0.5x)
            for d, od in zip(distances, orig_distances):
                if od > 0:
                    scale = d / od
                    if scale > 2.5 or scale < 0.4:
                        print(f"    ⚠ Homography validation failed: extreme scaling ({scale:.2f})")
                        return False
            
            # Check if transformation causes extreme distortion
            # by comparing aspect ratios
            width_ratio = distances[0] / distances[2] if distances[2] > 0 else 1
            height_ratio = distances[1] / distances[3] if distances[3] > 0 else 1
            
            if abs(width_ratio - 1.0) > 0.3 or abs(height_ratio - 1.0) > 0.3:
                print(f"    ⚠ Homography validation failed: aspect distortion")
                return False
            
            return True
            
        except Exception as e:
            print(f"    ⚠ Homography validation error: {e}")
            return False

    def stitch_with_homography(self, img1, img2, kp1, kp2, matches):
        """Stitch images using homography transformation with proper blending."""
        if len(matches) < 4:
            return None
        
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find homography with RANSAC
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
        
        # Validate homography
        if not self.validate_homography(H, img1.shape, img2.shape):
            print(f"    Homography rejected - transformation too extreme")
            return None
        
        # Calculate inlier ratio
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches) if len(matches) > 0 else 0
        
        print(f"    Inliers: {inliers}/{len(matches)} ({inlier_ratio:.2%})")
        
        # Reject if inlier ratio is too low
        if inlier_ratio < 0.3:
            print(f"    Homography rejected - low inlier ratio")
            return None
        
        # Calculate output size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of img2
        corners2 = np.float32([
            [0, 0], [0, h2], [w2, h2], [w2, 0]
        ]).reshape(-1, 1, 2)
        
        # Transform corners
        corners2_transformed = cv2.perspectiveTransform(corners2, H)
        
        # Combine with img1 corners
        corners1 = np.float32([
            [0, 0], [0, h1], [w1, h1], [w1, 0]
        ]).reshape(-1, 1, 2)
        
        all_corners = np.concatenate([corners1, corners2_transformed], axis=0)
        
        # Find bounding box
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Adjust homography for translation
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        
        # Warp img2
        result = cv2.warpPerspective(
            img2, translation.dot(H),
            (x_max - x_min, y_max - y_min)
        )
        
        # Create a mask for img1
        img1_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        img1_y_start = -y_min
        img1_y_end = -y_min + h1
        img1_x_start = -x_min
        img1_x_end = -x_min + w1
        
        img1_mask[img1_y_start:img1_y_end, img1_x_start:img1_x_end] = 255
        
        # Create a mask for warped img2
        img2_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, img2_mask = cv2.threshold(img2_gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find overlap region
        overlap_mask = cv2.bitwise_and(img1_mask, img2_mask)
        
        # Blend in overlap region
        if np.sum(overlap_mask) > 100:  # If there's significant overlap
            # Create distance transforms for smooth blending
            dist1 = cv2.distanceTransform(img1_mask, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(img2_mask, cv2.DIST_L2, 5)
            
            # Normalize distances
            dist_sum = dist1 + dist2
            dist_sum[dist_sum == 0] = 1  # Avoid division by zero
            
            alpha1 = (dist1 / dist_sum).astype(np.float32)
            alpha2 = (dist2 / dist_sum).astype(np.float32)
            
            # Expand alpha to 3 channels
            alpha1 = np.expand_dims(alpha1, axis=2)
            alpha2 = np.expand_dims(alpha2, axis=2)
            
            # Place img1 in result with blending
            img1_padded = np.zeros_like(result, dtype=np.float32)
            img1_padded[img1_y_start:img1_y_end, img1_x_start:img1_x_end] = img1.astype(np.float32)
            
            # Blend
            result = (result.astype(np.float32) * alpha2 + img1_padded * alpha1).astype(np.uint8)
        else:
            # No significant overlap, just place img1
            result[img1_y_start:img1_y_end, img1_x_start:img1_x_end] = img1
        
        return result

    def should_use_simple_stitch(self, best_config):
        """Determine if images should use simple edge-based stitching."""
        # If both images have no rotation and no flip, use simple stitch
        no_transform = (
            best_config['img1_rot'] == 0 and 
            best_config['img1_flip'] is None and
            best_config['img2_rot'] == 0 and 
            best_config['img2_flip'] is None
        )
        
        # If edges suggest simple adjacent placement
        simple_edges = (
            (best_config['edge1'] == 'bottom' and best_config['edge2'] == 'top') or
            (best_config['edge1'] == 'top' and best_config['edge2'] == 'bottom') or
            (best_config['edge1'] == 'right' and best_config['edge2'] == 'left') or
            (best_config['edge1'] == 'left' and best_config['edge2'] == 'right')
        )
        
        return no_transform and simple_edges

    def stitch_with_alignment(self, img1, img2, edge1, edge2):
        """Stitch images based on edge alignment."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Determine stitching direction
        if edge1 == 'bottom' and edge2 == 'top':
            # Stack vertically: img1 on top, img2 on bottom
            return self.blend_vertical(img1, img2, 'bottom')
        elif edge1 == 'top' and edge2 == 'bottom':
            # Stack vertically: img2 on top, img1 on bottom
            return self.blend_vertical(img2, img1, 'bottom')
        elif edge1 == 'right' and edge2 == 'left':
            # Stack horizontally: img1 on left, img2 on right
            return self.blend_horizontal(img1, img2, 'right')
        elif edge1 == 'left' and edge2 == 'right':
            # Stack horizontally: img2 on left, img1 on right
            return self.blend_horizontal(img2, img1, 'right')
        
        return None

    def blend_vertical(self, img_top, img_bottom, join_edge):
        """Blend two images vertically with smooth transition."""
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        # Match widths
        target_width = max(w1, w2)
        img_top_resized = cv2.resize(
            img_top, (target_width, int(h1 * target_width / w1))
        )
        img_bottom_resized = cv2.resize(
            img_bottom, (target_width, int(h2 * target_width / w2))
        )
        
        h1_new, h2_new = img_top_resized.shape[0], img_bottom_resized.shape[0]
        
        # Create blending region
        blend_height = min(100, h1_new // 8, h2_new // 8)
        
        if blend_height > 5:
            # Extract overlap regions
            bottom_strip = img_top_resized[-blend_height:, :].astype(float)
            top_strip = img_bottom_resized[:blend_height, :].astype(float)
            
            # Create smooth alpha mask
            alpha = np.linspace(1, 0, blend_height).reshape(-1, 1, 1)
            alpha = np.power(alpha, 0.5)  # Non-linear for smoother transition
            
            # Blend
            blended = (bottom_strip * alpha + top_strip * (1 - alpha))
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Combine
            result = np.vstack([
                img_top_resized[:-blend_height, :],
                blended,
                img_bottom_resized[blend_height:, :]
            ])
        else:
            result = np.vstack([img_top_resized, img_bottom_resized])
        
        return result

    def blend_horizontal(self, img_left, img_right, join_edge):
        """Blend two images horizontally with smooth transition."""
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        
        # Match heights
        target_height = max(h1, h2)
        img_left_resized = cv2.resize(
            img_left, (int(w1 * target_height / h1), target_height)
        )
        img_right_resized = cv2.resize(
            img_right, (int(w2 * target_height / h2), target_height)
        )
        
        w1_new, w2_new = img_left_resized.shape[1], img_right_resized.shape[1]
        
        # Create blending region
        blend_width = min(100, w1_new // 8, w2_new // 8)
        
        if blend_width > 5:
            # Extract overlap regions
            right_strip = img_left_resized[:, -blend_width:].astype(float)
            left_strip = img_right_resized[:, :blend_width].astype(float)
            
            # Create smooth alpha mask
            alpha = np.linspace(1, 0, blend_width).reshape(1, -1, 1)
            alpha = np.power(alpha, 0.5)  # Non-linear for smoother transition
            
            # Blend
            blended = (right_strip * alpha + left_strip * (1 - alpha))
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Combine
            result = np.hstack([
                img_left_resized[:, :-blend_width],
                blended,
                img_right_resized[:, blend_width:]
            ])
        else:
            result = np.hstack([img_left_resized, img_right_resized])
        
        return result

    def reconstruct(
        self, image1_path, image2_path, output_path="reconstructed.jpg"
    ):
        """Main reconstruction method."""
        print(f"\n{'='*70}")
        print(f"ADVANCED IMAGE RECONSTRUCTION - TORN IMAGE REPAIR")
        print(f"{'='*70}\n")
        
        print(f"Loading images...")
        img1, img2 = self.load_images(image1_path, image2_path)
        print(f"  Image 1: {img1.shape}")
        print(f"  Image 2: {img2.shape}")

        # Step 1: Find best configuration (rotation, flip, edge matching)
        print(f"\nStep 1: Finding optimal alignment...")
        best_config = self.find_best_configuration(img1, img2)
        
        if best_config['score'] < 0.3:
            print(f"\n⚠ Warning: Low confidence score ({best_config['score']:.3f})")
            print(f"  Results may not be accurate.\n")
        
        img1_aligned = best_config['img1_trans']
        img2_aligned = best_config['img2_trans']
        
        # Check if we should use simple stitching
        use_simple = self.should_use_simple_stitch(best_config)
        
        if use_simple:
            print(f"\n  Using simple edge-based stitching (no rotation detected)")
        
        # Step 2: Try feature-based stitching (only if complex transformation needed)
        print(f"\nStep 2: Attempting feature-based stitching...")
        result = None
        method = "correlation"
        match_count = 0
        
        if not use_simple:
            try:
                kp1, des1, kp2, des2, feature_method = \
                    self.detect_features_robust(img1_aligned, img2_aligned)
                
                print(f"  Method: {feature_method}")
                print(f"  Keypoints - Image1: {len(kp1)}, Image2: {len(kp2)}")
                
                if des1 is not None and des2 is not None:
                    matches = self.match_features_robust(des1, des2, feature_method)
                    match_count = len(matches)
                    print(f"  Matches found: {match_count}")
                    
                    if match_count >= 10:
                        print(f"  Attempting homography-based stitching...")
                        result = self.stitch_with_homography(
                            img1_aligned, img2_aligned, kp1, kp2, matches
                        )
                        
                        if result is not None:
                            method = f"homography_{feature_method}"
                            print(f"  ✓ Homography stitching successful!")
            except Exception as e:
                print(f"  Feature-based stitching failed: {e}")
        else:
            print(f"  Skipping feature-based stitching (simple alignment detected)")
        
        # Step 3: Fall back to edge-based stitching
        if result is None:
            print(f"\nStep 3: Using edge-based stitching...")
            result = self.stitch_with_alignment(
                img1_aligned, img2_aligned,
                best_config['edge1'], best_config['edge2']
            )
            method = "edge_alignment"
            
            if result is None:
                print(f"  ✗ Edge-based stitching failed")
                raise ValueError("Could not reconstruct image")
        
        # Step 4: Post-processing
        print(f"\nStep 4: Post-processing...")
        
        # Remove black borders
        result = self.remove_black_borders(result)
        
        # Optional: Enhance result
        result = self.enhance_result(result)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS! Reconstruction complete")
        print(f"  Method: {method}")
        print(f"  Confidence: {best_config['score']:.3f}")
        print(f"  Output size: {result.shape}")
        print(f"  Saved to: {output_path}")
        print(f"{'='*70}\n")

        # Save to database
        self.save_to_database(
            image1_path, image2_path, output_path, "success",
            method, match_count, match_count, 0
        )

        return result

    def remove_black_borders(self, img, threshold=10):
        """Remove black borders from stitched image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return img
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop with small margin
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        return img[y:y+h, x:x+w]

    def enhance_result(self, img):
        """Enhance the reconstructed image."""
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(img, 5, 50, 50)
        
        # Slight sharpening
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]]) / 1.0
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced

    def save_to_database(
        self, img1_path, img2_path, output_path, status, method,
        keypoints, matches, rotation
    ):
        """Save reconstruction information to database."""
        self.cursor.execute(
            """
            INSERT INTO reconstructions
            (image1_path, image2_path, output_path, status, method, 
             rotation_detected)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (img1_path, img2_path, output_path, status, method, rotation),
        )

        reconstruction_id = self.cursor.lastrowid

        self.cursor.execute(
            """
            INSERT INTO reconstruction_metadata
            (reconstruction_id, keypoints_found, matches_found, confidence)
            VALUES (?, ?, ?, ?)
        """,
            (reconstruction_id, keypoints, matches, 0),
        )

        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def print_usage():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("ADVANCED IMAGE RECONSTRUCTION TOOL - TORN IMAGE REPAIR")
    print("=" * 70)
    print("\nCapabilities:")
    print("  ✓ Handles tears from any side (top, bottom, left, right)")
    print("  ✓ Detects and corrects rotations (0°, 90°, 180°, 270°)")
    print("  ✓ Handles flipped/mirrored images")
    print("  ✓ Works with arbitrary tear angles (uses homography)")
    print("  ✓ Automatic edge matching and alignment")
    print("  ✓ Smart blending for seamless reconstruction")
    print("\nUsage:")
    print("  python3 improved_reconstructor.py <image1> <image2> [output]")
    print("\nExample:")
    print("  python3 improved_reconstructor.py part1.jpg part2.jpg result.jpg")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        print("ERROR: Please provide two image paths!\n")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "reconstructed.jpg"

    reconstructor = AdvancedImageReconstructor()

    try:
        result = reconstructor.reconstruct(
            image1_path, image2_path, output_path
        )
        print("Reconstruction completed successfully!")
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        reconstructor.close()
