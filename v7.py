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
                rotation_detected REAL,
                img1_rotation TEXT,
                img2_rotation TEXT
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

    def generate_orientations(self, img):
        """Generate all possible orientations of an image.
        
        Returns dict with keys: 0, 90, 180, 270, flip_h, flip_v
        """
        orientations = {
            '0': img.copy(),
            '90': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            '180': cv2.rotate(img, cv2.ROTATE_180),
            '270': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            'flip_h': cv2.flip(img, 1),  # Horizontal flip
            'flip_v': cv2.flip(img, 0),  # Vertical flip (inverted)
        }
        return orientations

    def analyze_image_content(self, img):
        """Analyze image to determine which part (top/bottom) has more detail."""
        h, w = img.shape[:2]
        
        if h < 10 or w < 10:
            return None
        
        # Split into top and bottom halves
        top_half = img[:h//2, :]
        bottom_half = img[h//2:, :]
        
        # Calculate variance (more detail = higher variance)
        top_var = np.var(top_half)
        bottom_var = np.var(bottom_half)
        
        # Calculate mean brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        top_gray = gray[:h//2, :]
        bottom_gray = gray[h//2:, :]
        
        top_mean = np.mean(top_gray)
        bottom_mean = np.mean(bottom_gray)
        
        # Adaptive edge detection
        sigma = 0.33
        median_val = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        
        edges = cv2.Canny(gray, lower, upper)
        top_edges = edges[:h//2, :]
        bottom_edges = edges[h//2:, :]
        
        top_edge_density = np.sum(top_edges > 0) / top_edges.size
        bottom_edge_density = np.sum(bottom_edges > 0) / bottom_edges.size
        
        return {
            'top_var': top_var,
            'bottom_var': bottom_var,
            'top_mean': top_mean,
            'bottom_mean': bottom_mean,
            'top_edge_density': top_edge_density,
            'bottom_edge_density': bottom_edge_density,
            'height': h,
            'width': w
        }

    def calculate_match_score(self, img1_info, img2_info):
        """Calculate how well img1-bottom matches img2-top.
        
        Returns normalized score (higher is better).
        """
        if img1_info is None or img2_info is None:
            return -1000
        
        # Normalize brightness difference (0-1, higher is better)
        brightness_diff = abs(img1_info['bottom_mean'] - img2_info['top_mean'])
        brightness_score = 1.0 - (brightness_diff / 255.0)
        
        # Edge density matching (buildings at bottom of img1, less at top of img2)
        edge_score = 0
        if img1_info['bottom_edge_density'] > img2_info['top_edge_density']:
            edge_ratio = img2_info['top_edge_density'] / (img1_info['bottom_edge_density'] + 0.001)
            edge_score = 1.0 - edge_ratio
        else:
            edge_score = -0.5  # Penalty if reversed
        
        # Variance similarity at boundary
        var_diff = abs(img1_info['bottom_var'] - img2_info['top_var'])
        var_score = 1.0 / (1.0 + var_diff / 10000.0)
        
        # Width similarity (should be close for panoramas)
        width_ratio = min(img1_info['width'], img2_info['width']) / max(img1_info['width'], img2_info['width'])
        
        # Combined score with weights
        total_score = (
            brightness_score * 0.35 +
            edge_score * 0.35 +
            var_score * 0.15 +
            width_ratio * 0.15
        )
        
        return total_score

    def find_best_orientation(self, img1, img2):
        """Find the best orientation combination for both images.
        
        Returns: (best_img1, best_img2, img1_transform, img2_transform, score)
        """
        print(f"\n  🔍 Testing all orientation combinations...")
        
        orientations1 = self.generate_orientations(img1)
        orientations2 = self.generate_orientations(img2)
        
        best_score = -999999
        best_config = None
        
        results = []
        
        # Test all combinations
        for o1_name, o1_img in orientations1.items():
            info1 = self.analyze_image_content(o1_img)
            if info1 is None:
                continue
                
            for o2_name, o2_img in orientations2.items():
                info2 = self.analyze_image_content(o2_img)
                if info2 is None:
                    continue
                
                # Test img1 on top, img2 on bottom
                score_1_2 = self.calculate_match_score(info1, info2)
                
                # Test img2 on top, img1 on bottom
                score_2_1 = self.calculate_match_score(info2, info1)
                
                if score_1_2 > best_score:
                    best_score = score_1_2
                    best_config = (o1_img, o2_img, o1_name, o2_name, False, score_1_2)
                
                if score_2_1 > best_score:
                    best_score = score_2_1
                    best_config = (o2_img, o1_img, o2_name, o1_name, True, score_2_1)
                
                results.append({
                    'config': f"{o1_name}→{o2_name}",
                    'score_1_2': score_1_2,
                    'score_2_1': score_2_1
                })
        
        # Show top 5 results
        results.sort(key=lambda x: max(x['score_1_2'], x['score_2_1']), reverse=True)
        print(f"\n  📊 Top 5 configurations:")
        for i, r in enumerate(results[:5], 1):
            best_dir = "→" if r['score_1_2'] > r['score_2_1'] else "←"
            best_s = max(r['score_1_2'], r['score_2_1'])
            print(f"     {i}. {r['config']} {best_dir} Score: {best_s:.3f}")
        
        if best_config is None:
            raise ValueError("Could not find valid orientation combination")
        
        img_top, img_bottom, o1_name, o2_name, swapped, score = best_config
        
        print(f"\n  ✅ BEST MATCH (score: {score:.3f}):")
        if swapped:
            print(f"     Image 2 ({o2_name}) on TOP")
            print(f"     Image 1 ({o1_name}) on BOTTOM")
        else:
            print(f"     Image 1 ({o1_name}) on TOP")
            print(f"     Image 2 ({o2_name}) on BOTTOM")
        
        return img_top, img_bottom, o1_name, o2_name, swapped, score

    def find_horizontal_offset(self, strip1, strip2):
        """Find the best horizontal offset between two strips."""
        h1, w1 = strip1.shape[:2]
        h2, w2 = strip2.shape[:2]
        
        best_offset = 0
        best_score = float('inf')
        
        # Try different offsets (proportional to image width)
        max_offset = int(min(w1, w2) * 0.25)  # Search up to 25% of width
        step = max(1, max_offset // 100)
        
        for offset in range(-max_offset, max_offset, step):
            # Calculate overlap region
            if offset >= 0:
                overlap_w = min(w1 - offset, w2)
                if overlap_w <= 50:
                    continue
                s1 = strip1[:, offset:offset + overlap_w]
                s2 = strip2[:, :overlap_w]
            else:
                overlap_w = min(w1, w2 + offset)
                if overlap_w <= 50:
                    continue
                s1 = strip1[:, :overlap_w]
                s2 = strip2[:, -offset:-offset + overlap_w]
            
            # Ensure same size
            min_h = min(s1.shape[0], s2.shape[0])
            min_w = min(s1.shape[1], s2.shape[1])
            s1 = s1[:min_h, :min_w]
            s2 = s2[:min_h, :min_w]
            
            if s1.size == 0 or s2.size == 0:
                continue
            
            # Calculate difference using MSE
            diff = cv2.absdiff(s1, s2)
            score = np.mean(diff)
            
            if score < best_score:
                best_score = score
                best_offset = offset
        
        return best_offset, best_score

    def stitch_vertical_with_offset(self, img_top, img_bottom):
        """Stitch images vertically with horizontal offset correction."""
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        print(f"\n  🔧 Finding horizontal alignment...")
        
        # Use bottom strip of top image and top strip of bottom image
        strip_height = min(100, h1 // 4, h2 // 4)
        
        strip_top = img_top[-strip_height:, :]
        strip_bottom = img_bottom[:strip_height, :]
        
        # Find best horizontal offset
        offset, score = self.find_horizontal_offset(strip_top, strip_bottom)
        
        print(f"     Horizontal offset: {offset}px")
        print(f"     Alignment quality: {max(0, 255 - score):.1f}/255")
        
        # Validate alignment
        if score > 100:
            print(f"     ⚠️  Warning: High difference score - images may be unrelated")
        
        # Create canvas
        canvas_w = max(w1, w2 + abs(offset))
        canvas_h = h1 + h2 - strip_height  # Overlap the strips
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Calculate placements
        if offset < 0:
            x_offset_top = abs(offset)
            x_offset_bottom = 0
        else:
            x_offset_top = 0
            x_offset_bottom = offset
        
        # Place top image
        canvas[:h1, x_offset_top:x_offset_top + w1] = img_top
        
        # Blend zone
        blend_start = h1 - strip_height
        blend_end = h1
        
        # Calculate overlap region
        overlap_x_start = max(x_offset_top, x_offset_bottom)
        overlap_x_end = min(x_offset_top + w1, x_offset_bottom + w2)
        
        if overlap_x_end > overlap_x_start and blend_start >= 0:
            # Gradient blending
            for i in range(strip_height):
                alpha = i / strip_height  # 0 at top, 1 at bottom
                y_pos = blend_start + i
                
                if y_pos >= 0 and y_pos < canvas_h:
                    # Get pixels from both images
                    top_x1 = overlap_x_start - x_offset_top
                    top_x2 = overlap_x_end - x_offset_top
                    bot_x1 = overlap_x_start - x_offset_bottom
                    bot_x2 = overlap_x_end - x_offset_bottom
                    
                    if (0 <= top_x1 < w1 and 0 <= top_x2 <= w1 and 
                        0 <= bot_x1 < w2 and 0 <= bot_x2 <= w2 and
                        i < h2):
                        
                        top_pixels = img_top[h1 - strip_height + i, top_x1:top_x2]
                        bot_pixels = img_bottom[i, bot_x1:bot_x2]
                        
                        # Ensure same width
                        min_width = min(top_pixels.shape[0], bot_pixels.shape[0])
                        if min_width > 0:
                            blended = cv2.addWeighted(
                                top_pixels[:min_width].astype(float), 1 - alpha,
                                bot_pixels[:min_width].astype(float), alpha,
                                0
                            )
                            canvas[y_pos, overlap_x_start:overlap_x_start + min_width] = blended.astype(np.uint8)
        
        # Place remaining part of bottom image
        remaining_start = h1
        remaining_bottom = img_bottom[strip_height:, :]
        canvas[remaining_start:remaining_start + remaining_bottom.shape[0],
               x_offset_bottom:x_offset_bottom + w2] = remaining_bottom
        
        # Crop empty regions
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            canvas = canvas[y:y+h, x:x+w]
        
        return canvas, offset

    def reconstruct(
        self, image1_path, image2_path, output_path="reconstructed.jpg"
    ):
        """Main reconstruction method with rotation/inversion detection."""
        print(f"\n{'='*70}")
        print(f"🔄 ADVANCED IMAGE RECONSTRUCTION WITH ROTATION DETECTION")
        print(f"{'='*70}\n")
        
        print(f"📂 Loading images...")
        img1, img2 = self.load_images(image1_path, image2_path)
        print(f"   Image 1: {img1.shape[1]}x{img1.shape[0]}")
        print(f"   Image 2: {img2.shape[1]}x{img2.shape[0]}")
        
        # Find best orientation
        img_top, img_bottom, o1_transform, o2_transform, swapped, match_score = \
            self.find_best_orientation(img1, img2)
        
        # Stitch with alignment
        result, offset = self.stitch_vertical_with_offset(img_top, img_bottom)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! Reconstruction complete")
        print(f"{'='*70}")
        print(f"📁 Output: {output_path}")
        print(f"📐 Size: {result.shape[1]}x{result.shape[0]}")
        print(f"🔄 Transformations applied:")
        if swapped:
            print(f"   • Image 2: {o2_transform} (TOP)")
            print(f"   • Image 1: {o1_transform} (BOTTOM)")
        else:
            print(f"   • Image 1: {o1_transform} (TOP)")
            print(f"   • Image 2: {o2_transform} (BOTTOM)")
        print(f"↔️  Horizontal alignment: {offset}px")
        print(f"⭐ Match confidence: {match_score:.3f}")
        print(f"{'='*70}\n")
        
        # Save to database
        self.save_to_database(
            image1_path, image2_path, output_path,
            "success_with_rotation", "orientation_detection",
            0, 0, 0, o1_transform, o2_transform
        )
        
        return result

    def save_to_database(
        self, img1_path, img2_path, output_path, status, method,
        keypoints, matches, rotation, img1_rot, img2_rot
    ):
        """Save reconstruction information to database."""
        self.cursor.execute(
            """
            INSERT INTO reconstructions
            (image1_path, image2_path, output_path, status, method, 
             rotation_detected, img1_rotation, img2_rotation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (img1_path, img2_path, output_path, status, method, rotation,
             img1_rot, img2_rot),
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
    print("🔄 ADVANCED IMAGE RECONSTRUCTION TOOL")
    print("With automatic rotation/inversion detection")
    print("=" * 70)
    print("\nUsage:")
    print("  python3 v1.py <image1> <image2> [output]")
    print("\nFeatures:")
    print("  ✅ Detects 0°, 90°, 180°, 270° rotations")
    print("  ✅ Detects horizontal/vertical flips")
    print("  ✅ Tests all orientation combinations")
    print("  ✅ Automatic image ordering")
    print("  ✅ Horizontal offset correction")
    print("  ✅ Intelligent seam blending")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        print("❌ ERROR: Please provide two image paths!\n")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "reconstructed.jpg"

    reconstructor = AdvancedImageReconstructor()

    try:
        result = reconstructor.reconstruct(
            image1_path, image2_path, output_path
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        reconstructor.close()
