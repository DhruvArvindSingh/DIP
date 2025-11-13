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
        self.debug_mode = True  # Enable debug output

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

    def save_debug_image(self, img, name):
        """Save debug visualization."""
        if self.debug_mode:
            cv2.imwrite(f"debug_{name}.jpg", img)

    def generate_orientations(self, img):
        """Generate all possible orientations of an image."""
        orientations = {
            '0': img.copy(),
            '90': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            '180': cv2.rotate(img, cv2.ROTATE_180),
            '270': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            'flip_h': cv2.flip(img, 1),
            'flip_v': cv2.flip(img, 0),
        }
        return orientations

    def detect_horizon(self, img):
        """Detect the horizon line in the image using edge detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 30, 100)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Find horizontal lines (low angle)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Nearly horizontal
                horizontal_lines.append((y1 + y2) / 2)
        
        if horizontal_lines:
            # Return median y-position of horizontal lines
            return np.median(horizontal_lines)
        
        return None

    def analyze_image_content(self, img):
        """Comprehensive image analysis."""
        h, w = img.shape[:2]
        
        if h < 10 or w < 10:
            return None
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        is_landscape = aspect_ratio > 1.2
        is_portrait = aspect_ratio < 0.8
        
        # Split into regions
        top_quarter = img[:h//4, :]
        top_half = img[:h//2, :]
        bottom_half = img[h//2:, :]
        bottom_quarter = img[3*h//4:, :]
        
        # Analyze sky characteristics (usually in top)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        top_brightness = np.mean(gray[:h//3, :])
        middle_brightness = np.mean(gray[h//3:2*h//3, :])
        bottom_brightness = np.mean(gray[2*h//3:, :])
        
        # Variance (detail)
        top_var = np.var(gray[:h//2, :])
        bottom_var = np.var(gray[h//2:, :])
        
        # Adaptive edge detection
        sigma = 0.33
        median_val = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        
        edges = cv2.Canny(gray, lower, upper)
        
        # Edge density per region
        top_edges = np.sum(edges[:h//3, :] > 0) / (edges[:h//3, :].size)
        middle_edges = np.sum(edges[h//3:2*h//3, :] > 0) / (edges[h//3:2*h//3, :].size)
        bottom_edges = np.sum(edges[2*h//3:, :] > 0) / (edges[2*h//3:, :].size)
        
        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Sky usually has high saturation in sunset
        top_saturation = np.mean(hsv[:h//3, :, 1])
        bottom_saturation = np.mean(hsv[2*h//3:, :, 1])
        
        # Detect horizon
        horizon_y = self.detect_horizon(img)
        
        # Sky score (bright, high saturation, low edges)
        top_sky_score = (top_brightness / 255) * 0.4 + \
                        (top_saturation / 255) * 0.4 + \
                        (1 - top_edges / 0.1) * 0.2
        
        bottom_sky_score = (bottom_brightness / 255) * 0.4 + \
                           (bottom_saturation / 255) * 0.4 + \
                           (1 - bottom_edges / 0.1) * 0.2
        
        return {
            'height': h,
            'width': w,
            'aspect_ratio': aspect_ratio,
            'is_landscape': is_landscape,
            'is_portrait': is_portrait,
            'top_brightness': top_brightness,
            'middle_brightness': middle_brightness,
            'bottom_brightness': bottom_brightness,
            'top_var': top_var,
            'bottom_var': bottom_var,
            'top_edges': top_edges,
            'middle_edges': middle_edges,
            'bottom_edges': bottom_edges,
            'top_saturation': top_saturation,
            'bottom_saturation': bottom_saturation,
            'top_sky_score': top_sky_score,
            'bottom_sky_score': bottom_sky_score,
            'horizon_y': horizon_y
        }

    def match_features(self, img1, img2):
        """Use feature matching to validate orientation."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=500)
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0, []
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Return match count and good matches
        good_matches = [m for m in matches if m.distance < 50]
        
        return len(good_matches), good_matches

    def calculate_match_score(self, img1, img2, info1, info2):
        """Calculate comprehensive match score for img1-top, img2-bottom."""
        if info1 is None or info2 is None:
            return -1000
        
        score = 0
        details = {}
        
        # 1. Aspect ratio check (both should be landscape for panorama)
        aspect_penalty = 0
        if not info1['is_landscape'] or not info2['is_landscape']:
            aspect_penalty = -0.3
        details['aspect'] = -aspect_penalty
        
        # 2. Brightness gradient (sky bright on top, ground darker below)
        # For sunset: top of img1 should be bright, bottom of img2 should be darker
        brightness_gradient = (info1['top_brightness'] - info2['bottom_brightness']) / 255
        brightness_score = max(0, brightness_gradient) * 0.3
        details['brightness_gradient'] = brightness_score
        
        # 3. Edge density (more edges at bottom = buildings/ground)
        edge_ratio = 0
        if info1['bottom_edges'] > 0.01 and info2['top_edges'] > 0.01:
            # Similar edge density at boundary = good match
            edge_similarity = 1 - abs(info1['bottom_edges'] - info2['top_edges']) / max(info1['bottom_edges'], info2['top_edges'])
            edge_ratio = edge_similarity * 0.2
        details['edge_match'] = edge_ratio
        
        # 4. Color continuity at boundary
        color_match = 1 - abs(info1['bottom_brightness'] - info2['top_brightness']) / 255
        color_score = color_match * 0.25
        details['color_continuity'] = color_score
        
        # 5. Sky detection (sky should be on top)
        if info1['top_sky_score'] > 0.5 and info2['bottom_sky_score'] < 0.5:
            sky_score = 0.15  # Correct: sky on top
        elif info1['bottom_sky_score'] > 0.5 and info2['top_sky_score'] > 0.5:
            sky_score = -0.2  # Wrong: sky on both ends
        else:
            sky_score = 0
        details['sky_position'] = sky_score
        
        # 6. Width similarity
        width_ratio = min(info1['width'], info2['width']) / max(info1['width'], info2['width'])
        width_score = width_ratio * 0.1
        details['width_match'] = width_score
        
        # 7. Feature matching validation
        num_matches, _ = self.match_features(img1, img2)
        feature_score = min(num_matches / 100, 0.2)  # Cap at 0.2
        details['features'] = feature_score
        
        # Total score
        score = (brightness_score + edge_ratio + color_score + 
                sky_score + width_score + feature_score + aspect_penalty)
        
        details['total'] = score
        
        return score, details

    def find_best_orientation(self, img1, img2):
        """Find the best orientation with detailed analysis."""
        print(f"\n  🔍 Testing orientation combinations...")
        
        orientations1 = self.generate_orientations(img1)
        orientations2 = self.generate_orientations(img2)
        
        best_score = -999999
        best_config = None
        best_details = None
        
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
                score_1_2, details_1_2 = self.calculate_match_score(
                    o1_img, o2_img, info1, info2
                )
                
                # Test img2 on top, img1 on bottom
                score_2_1, details_2_1 = self.calculate_match_score(
                    o2_img, o1_img, info2, info1
                )
                
                if score_1_2 > best_score:
                    best_score = score_1_2
                    best_config = (o1_img, o2_img, o1_name, o2_name, False)
                    best_details = details_1_2
                
                if score_2_1 > best_score:
                    best_score = score_2_1
                    best_config = (o2_img, o1_img, o2_name, o1_name, True)
                    best_details = details_2_1
                
                results.append({
                    'config': f"Img1:{o1_name} → Img2:{o2_name}",
                    'score': score_1_2,
                    'ar1': info1['aspect_ratio'],
                    'ar2': info2['aspect_ratio']
                })
                
                results.append({
                    'config': f"Img2:{o2_name} → Img1:{o1_name}",
                    'score': score_2_1,
                    'ar1': info2['aspect_ratio'],
                    'ar2': info1['aspect_ratio']
                })
        
        # Show top 10 results
        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"\n  📊 Top 10 configurations:")
        for i, r in enumerate(results[:10], 1):
            print(f"     {i:2d}. {r['config']:30s} | Score: {r['score']:6.3f} | AR: {r['ar1']:.2f}, {r['ar2']:.2f}")
        
        if best_config is None:
            raise ValueError("Could not find valid orientation combination")
        
        img_top, img_bottom, o1_name, o2_name, swapped = best_config
        
        print(f"\n  ✅ SELECTED (score: {best_score:.3f}):")
        if swapped:
            print(f"     TOP:    Image 2 ({o2_name})")
            print(f"     BOTTOM: Image 1 ({o1_name})")
        else:
            print(f"     TOP:    Image 1 ({o1_name})")
            print(f"     BOTTOM: Image 2 ({o2_name})")
        
        print(f"\n  📋 Score breakdown:")
        for key, value in best_details.items():
            if key != 'total':
                print(f"     {key:20s}: {value:+.3f}")
        
        # Save debug images
        self.save_debug_image(img_top, "top_selected")
        self.save_debug_image(img_bottom, "bottom_selected")
        
        return img_top, img_bottom, o1_name, o2_name, swapped, best_score

    def find_horizontal_offset(self, strip1, strip2):
        """Find the best horizontal offset between two strips."""
        h1, w1 = strip1.shape[:2]
        h2, w2 = strip2.shape[:2]
        
        best_offset = 0
        best_score = float('inf')
        
        # Try different offsets
        max_offset = int(min(w1, w2) * 0.3)
        step = max(1, max_offset // 50)
        
        for offset in range(-max_offset, max_offset, step):
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
            
            min_h = min(s1.shape[0], s2.shape[0])
            min_w = min(s1.shape[1], s2.shape[1])
            s1 = s1[:min_h, :min_w]
            s2 = s2[:min_h, :min_w]
            
            if s1.size == 0 or s2.size == 0:
                continue
            
            # Use structural similarity
            diff = cv2.absdiff(s1, s2)
            score = np.mean(diff)
            
            if score < best_score:
                best_score = score
                best_offset = offset
        
        return best_offset, best_score

    def stitch_vertical_with_offset(self, img_top, img_bottom):
        """Stitch images vertically with blending."""
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        print(f"\n  🔧 Stitching images...")
        print(f"     Top:    {w1}x{h1}")
        print(f"     Bottom: {w2}x{h2}")
        
        # Find overlap region
        strip_height = min(50, h1 // 5, h2 // 5)
        
        strip_top = img_top[-strip_height:, :]
        strip_bottom = img_bottom[:strip_height, :]
        
        # Find horizontal offset
        offset, score = self.find_horizontal_offset(strip_top, strip_bottom)
        
        print(f"     Horizontal offset: {offset}px")
        print(f"     Match quality: {max(0, 255 - score):.1f}/255")
        
        # Create canvas
        canvas_w = max(w1, w2 + abs(offset))
        canvas_h = h1 + h2 - strip_height
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Calculate positions
        if offset < 0:
            x1, x2 = abs(offset), 0
        else:
            x1, x2 = 0, offset
        
        # Place images
        canvas[:h1, x1:x1 + w1] = img_top
        
        # Blend overlap
        blend_region = h1 - strip_height
        
        for i in range(strip_height):
            alpha = i / strip_height
            y = blend_region + i
            
            if y >= 0 and y < h1 and i < h2:
                x_start = max(x1, x2)
                x_end = min(x1 + w1, x2 + w2)
                
                if x_end > x_start:
                    top_strip = img_top[y, x_start-x1:x_end-x1]
                    bot_strip = img_bottom[i, x_start-x2:x_end-x2]
                    
                    min_w = min(len(top_strip), len(bot_strip))
                    if min_w > 0:
                        blended = cv2.addWeighted(
                            top_strip[:min_w].astype(float), 1 - alpha,
                            bot_strip[:min_w].astype(float), alpha,
                            0
                        )
                        canvas[y, x_start:x_start + min_w] = blended.astype(np.uint8)
        
        # Place bottom part
        canvas[h1:, x2:x2 + w2] = img_bottom[strip_height:, :]
        
        # Crop black borders
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            canvas = canvas[y:y+h, x:x+w]
        
        print(f"     Final size: {canvas.shape[1]}x{canvas.shape[0]}")
        
        return canvas, offset

    def reconstruct(
        self, image1_path, image2_path, output_path="reconstructed.jpg"
    ):
        """Main reconstruction method."""
        print(f"\n{'='*70}")
        print(f"🔄 ADVANCED IMAGE RECONSTRUCTION v2.0")
        print(f"{'='*70}\n")
        
        print(f"📂 Loading images...")
        img1, img2 = self.load_images(image1_path, image2_path)
        print(f"   Image 1: {img1.shape[1]}x{img1.shape[0]} (WxH)")
        print(f"   Image 2: {img2.shape[1]}x{img2.shape[0]} (WxH)")
        
        # Save originals for debug
        self.save_debug_image(img1, "original_img1")
        self.save_debug_image(img2, "original_img2")
        
        # Find best orientation
        img_top, img_bottom, o1_transform, o2_transform, swapped, match_score = \
            self.find_best_orientation(img1, img2)
        
        # Stitch
        result, offset = self.stitch_vertical_with_offset(img_top, img_bottom)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        print(f"\n{'='*70}")
        print(f"✅ RECONSTRUCTION COMPLETE")
        print(f"{'='*70}")
        print(f"📁 Saved to: {output_path}")
        print(f"📐 Final size: {result.shape[1]}x{result.shape[0]}")
        print(f"🔄 Transformations:")
        if swapped:
            print(f"   • Image 2: {o2_transform} (placed on TOP)")
            print(f"   • Image 1: {o1_transform} (placed on BOTTOM)")
        else:
            print(f"   • Image 1: {o1_transform} (placed on TOP)")
            print(f"   • Image 2: {o2_transform} (placed on BOTTOM)")
        print(f"⭐ Confidence: {match_score:.3f}")
        print(f"{'='*70}\n")
        
        if self.debug_mode:
            print(f"🔍 Debug images saved: debug_*.jpg\n")
        
        # Save to database
        self.save_to_database(
            image1_path, image2_path, output_path,
            "success", "multi_orientation_v2",
            0, 0, 0, o1_transform, o2_transform
        )
        
        return result

    def save_to_database(
        self, img1_path, img2_path, output_path, status, method,
        keypoints, matches, rotation, img1_rot, img2_rot
    ):
        """Save reconstruction info to database."""
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
    print("🔄 ADVANCED IMAGE RECONSTRUCTION v2.0")
    print("=" * 70)
    print("\nUsage:")
    print("  python3 v2.py <image1> <image2> [output]")
    print("\nFeatures:")
    print("  ✅ Automatic rotation detection (0°, 90°, 180°, 270°)")
    print("  ✅ Flip detection (horizontal/vertical)")
    print("  ✅ Aspect ratio validation")
    print("  ✅ Sky/ground detection")
    print("  ✅ Feature matching validation")
    print("  ✅ Horizon detection")
    print("  ✅ Debug visualization")
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
