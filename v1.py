import cv2
import numpy as np
import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime


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

    def preprocess_image(self, img):
        """Preprocess image for better feature detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def detect_features_robust(self, img1, img2):
        """Enhanced feature detection with multiple algorithms."""
        gray1 = self.preprocess_image(img1)
        gray2 = self.preprocess_image(img2)

        # Try SIFT first (best for rotation/scale invariance)
        try:
            sift = cv2.SIFT_create(nfeatures=10000)
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            method = "SIFT"
            print(f"✓ Using SIFT")
        except cv2.error:
            # Fall back to ORB
            orb = cv2.ORB_create(nfeatures=10000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            method = "ORB"
            print(f"✓ Using ORB")

        return kp1, des1, kp2, des2, method

    def match_features_robust(self, des1, des2, method):
        """Enhanced feature matching with filtering."""
        if method == "SIFT":
            # FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)  # Increased for better matching
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher for ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        matches = matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test with adaptive threshold
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Slightly relaxed
                    good_matches.append(m)

        return good_matches

    def estimate_rotation_angle(self, kp1, kp2, matches):
        """Estimate rotation angle between images."""
        if len(matches) < 3:
            return 0

        angles = []
        for match in matches[:20]:  # Use top 20 matches
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            angle1 = kp1[match.queryIdx].angle
            angle2 = kp2[match.trainIdx].angle
            
            angle_diff = angle2 - angle1
            angles.append(angle_diff)

        # Use median to be robust to outliers
        if angles:
            median_angle = np.median(angles)
            return median_angle
        return 0

    def rotate_image(self, img, angle):
        """Rotate image by given angle."""
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new dimensions
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        
        return rotated

    def align_images_with_homography(self, img1, img2, kp1, kp2, matches):
        """Align images using homography with RANSAC."""
        if len(matches) < 10:
            raise ValueError(
                f"Not enough matches: {len(matches)}. Need at least 10."
            )

        # Extract matched keypoints
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Find homography with RANSAC
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0
        )
        
        if H is None:
            raise ValueError("Could not compute homography")

        matches_mask = mask.ravel().tolist()
        inliers = sum(matches_mask)
        
        print(f"  Inliers: {inliers}/{len(matches)}")

        return H, inliers

    def stitch_images(self, img1, img2, H):
        """Stitch images using computed homography."""
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Get corners of both images
        corners1 = np.float32(
            [[0, 0], [0, height1], [width1, height1], [width1, 0]]
        ).reshape(-1, 1, 2)
        corners2 = np.float32(
            [[0, 0], [0, height2], [width2, height2], [width2, 0]]
        ).reshape(-1, 1, 2)

        # Transform corners of first image
        corners1_transformed = cv2.perspectiveTransform(corners1, H)

        # Combine all corners
        all_corners = np.concatenate((corners1_transformed, corners2), axis=0)

        # Calculate output canvas size
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Translation matrix
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Warp first image
        output_img = cv2.warpPerspective(
            img1,
            translation.dot(H),
            (x_max - x_min, y_max - y_min),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Place second image
        output_img[-y_min : height2 - y_min, -x_min : width2 - x_min] = img2

        return output_img

    def blend_images(self, img1, img2, H):
        """Advanced blending with feathering for seamless result."""
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Calculate canvas size
        corners1 = np.float32(
            [[0, 0], [0, height1], [width1, height1], [width1, 0]]
        ).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        
        corners2 = np.float32(
            [[0, 0], [0, height2], [width2, height2], [width2, 0]]
        ).reshape(-1, 1, 2)

        all_corners = np.concatenate((corners1_transformed, corners2), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Warp first image and create mask
        warped1 = cv2.warpPerspective(
            img1, translation.dot(H), (x_max - x_min, y_max - y_min)
        )
        mask1 = cv2.warpPerspective(
            np.ones_like(img1) * 255, translation.dot(H),
            (x_max - x_min, y_max - y_min)
        )

        # Create canvas for second image
        canvas2 = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
        mask2 = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
        
        canvas2[-y_min : height2 - y_min, -x_min : width2 - x_min] = img2
        mask2[-y_min : height2 - y_min, -x_min : width2 - x_min] = 255

        # Find overlap region
        overlap = cv2.bitwise_and(
            cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        )

        # Blend using weighted average in overlap region
        result = np.zeros_like(warped1)
        
        # Non-overlap regions
        result[mask1[:,:,0] > 0] = warped1[mask1[:,:,0] > 0]
        result[mask2[:,:,0] > 0] = canvas2[mask2[:,:,0] > 0]
        
        # Overlap region - blend with feathering
        overlap_mask = overlap > 0
        if np.any(overlap_mask):
            alpha = 0.5
            result[overlap_mask] = cv2.addWeighted(
                warped1[overlap_mask], alpha,
                canvas2[overlap_mask], 1 - alpha, 0
            )

        return result

    def try_multiple_orientations(self, img1, img2):
        """Try reconstructing with different orientations."""
        best_result = None
        best_matches = 0
        best_rotation = 0
        
        rotations = [0, 90, 180, 270]
        
        for rotation in rotations:
            print(f"\n  Trying rotation: {rotation}°")
            
            # Rotate img1
            if rotation != 0:
                img1_rotated = self.rotate_image(img1, rotation)
            else:
                img1_rotated = img1
            
            try:
                # Detect features
                kp1, des1, kp2, des2, method = self.detect_features_robust(
                    img1_rotated, img2
                )
                
                # Match features
                matches = self.match_features_robust(des1, des2, method)
                
                print(f"    Matches found: {len(matches)}")
                
                if len(matches) > best_matches:
                    best_matches = len(matches)
                    best_rotation = rotation
                    best_result = (img1_rotated, kp1, des1, kp2, des2, 
                                   matches, method)
                    
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        if best_result is None:
            raise ValueError("Could not find any valid orientation")
        
        print(f"\n✓ Best orientation: {best_rotation}° with {best_matches} matches")
        
        return best_result, best_rotation

    def reconstruct(
        self, image1_path, image2_path, output_path="reconstructed.jpg",
        try_rotations=True
    ):
        """Main reconstruction method with rotation handling."""
        print(f"\n{'='*60}")
        print(f"STARTING RECONSTRUCTION")
        print(f"{'='*60}\n")
        
        print(f"Loading images...")
        img1_orig, img2 = self.load_images(image1_path, image2_path)
        print(f"  Image 1: {img1_orig.shape}")
        print(f"  Image 2: {img2.shape}")

        rotation_angle = 0
        
        if try_rotations:
            print(f"\nTrying multiple orientations...")
            (img1, kp1, des1, kp2, des2, matches, method), rotation_angle = \
                self.try_multiple_orientations(img1_orig, img2)
        else:
            print(f"\nDetecting features...")
            kp1, des1, kp2, des2, method = self.detect_features_robust(
                img1_orig, img2
            )
            img1 = img1_orig
            
            print(f"\nMatching features...")
            matches = self.match_features_robust(des1, des2, method)
        
        print(f"\nMethod: {method}")
        print(f"Keypoints - Image1: {len(kp1)}, Image2: {len(kp2)}")
        print(f"Good matches: {len(matches)}")

        try:
            if len(matches) >= 10:
                print(f"\n✓ Stitching with homography...")
                
                # Compute homography
                H, inliers = self.align_images_with_homography(
                    img1, img2, kp1, kp2, matches
                )
                
                # Stitch images
                result = self.blend_images(img1, img2, H)
                
                # Crop black borders
                result = self.crop_black_borders(result)
                
                status = "success_homography"
                match_count = inliers
                
            elif len(matches) >= 4:
                print(f"\n⚠ Few matches. Attempting basic homography...")
                H, inliers = self.align_images_with_homography(
                    img1, img2, kp1, kp2, matches
                )
                result = self.stitch_images(img1, img2, H)
                result = self.crop_black_borders(result)
                status = "success_low_confidence"
                match_count = inliers
                
            else:
                print(f"\n⚠ Too few matches. Using concatenation...")
                result = self.simple_concatenate(img1, img2)
                status = "fallback_concatenation"
                match_count = len(matches)

            # Save result
            cv2.imwrite(output_path, result)
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS! Saved to: {output_path}")
            print(f"{'='*60}\n")

            # Save to database
            self.save_to_database(
                image1_path, image2_path, output_path, status, method,
                len(kp1) + len(kp2), match_count, rotation_angle
            )

            return result

        except Exception as e:
            print(f"\n✗ ERROR: {e}\n")
            status = "failed"
            self.save_to_database(
                image1_path, image2_path, output_path, status,
                "unknown", 0, 0, rotation_angle
            )
            raise

    def crop_black_borders(self, img, threshold=10):
        """Remove black borders from stitched image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return img[y:y+h, x:x+w]
        
        return img

    def simple_concatenate(self, img1, img2, direction="horizontal"):
        """Simple concatenation as fallback."""
        if direction == "horizontal":
            height = max(img1.shape[0], img2.shape[0])
            img1_resized = cv2.resize(
                img1, (int(img1.shape[1] * height / img1.shape[0]), height)
            )
            img2_resized = cv2.resize(
                img2, (int(img2.shape[1] * height / img2.shape[0]), height)
            )
            result = np.hstack((img1_resized, img2_resized))
        else:
            width = max(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(
                img1, (width, int(img1.shape[0] * width / img1.shape[1]))
            )
            img2_resized = cv2.resize(
                img2, (width, int(img2.shape[0] * width / img2.shape[1]))
            )
            result = np.vstack((img1_resized, img2_resized))
        
        return result

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
            (
                reconstruction_id,
                keypoints,
                matches,
                matches / max(keypoints, 1) if keypoints > 0 else 0,
            ),
        )

        self.conn.commit()

    def get_history(self, limit=10):
        """Get reconstruction history from database."""
        self.cursor.execute(
            """
            SELECT r.id, r.image1_path, r.image2_path, r.output_path,
                   r.created_at, r.status, r.method, r.rotation_detected,
                   m.matches_found, m.confidence
            FROM reconstructions r
            LEFT JOIN reconstruction_metadata m ON r.id = m.reconstruction_id
            ORDER BY r.created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        return self.cursor.fetchall()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def print_usage():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("ADVANCED IMAGE RECONSTRUCTION TOOL")
    print("=" * 60)
    print("\nUsage:")
    print("  python3 advanced_reconstructor.py <image1> <image2> [output]")
    print("\nExamples:")
    print("  python3 advanced_reconstructor.py part1.jpg part2.jpg")
    print("  python3 advanced_reconstructor.py left.png right.png result.jpg")
    print("\nFeatures:")
    print("  ✓ Automatic rotation detection")
    print("  ✓ Scale-invariant matching")
    print("  ✓ Advanced image blending")
    print("  ✓ Black border removal")
    print("=" * 60 + "\n")


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
            image1_path, image2_path, output_path, try_rotations=True
        )

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        sys.exit(1)
    finally:
        reconstructor.close()
