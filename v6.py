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

    def analyze_image_content(self, img):
        """Analyze image to determine which part (top/bottom) has more detail."""
        h, w = img.shape[:2]
        
        # Split into top and bottom halves
        top_half = img[:h//2, :]
        bottom_half = img[h//2:, :]
        
        # Calculate variance (more detail = higher variance)
        top_var = np.var(top_half)
        bottom_var = np.var(bottom_half)
        
        # Calculate mean brightness
        top_mean = np.mean(cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY))
        bottom_mean = np.mean(cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY))
        
        # Detect edges (buildings, objects)
        top_edges = cv2.Canny(cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY), 50, 150)
        bottom_edges = cv2.Canny(cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY), 50, 150)
        
        top_edge_density = np.sum(top_edges > 0) / top_edges.size
        bottom_edge_density = np.sum(bottom_edges > 0) / bottom_edges.size
        
        return {
            'top_var': top_var,
            'bottom_var': bottom_var,
            'top_mean': top_mean,
            'bottom_mean': bottom_mean,
            'top_edge_density': top_edge_density,
            'bottom_edge_density': bottom_edge_density
        }

    def determine_image_order(self, img1, img2):
        """Determine which image should be on top and which on bottom."""
        print(f"\n  Analyzing image content to determine order...")
        
        info1 = self.analyze_image_content(img1)
        info2 = self.analyze_image_content(img2)
        
        print(f"  Image 1 - Bottom edge density: {info1['bottom_edge_density']:.4f}")
        print(f"  Image 2 - Top edge density: {info2['top_edge_density']:.4f}")
        print(f"  Image 1 - Bottom brightness: {info1['bottom_mean']:.1f}")
        print(f"  Image 2 - Top brightness: {info2['top_mean']:.1f}")
        
        # Score for img1-top, img2-bottom configuration
        score_1_2 = 0
        
        # If img1 bottom has similar brightness to img2 top, they likely connect
        brightness_match_1_2 = abs(info1['bottom_mean'] - info2['top_mean'])
        score_1_2 -= brightness_match_1_2
        
        # If img1 bottom has more edges (buildings) and img2 top has fewer, good match
        if info1['bottom_edge_density'] > info2['top_edge_density']:
            score_1_2 += 100
        
        # Score for img2-top, img1-bottom configuration
        score_2_1 = 0
        brightness_match_2_1 = abs(info2['bottom_mean'] - info1['top_mean'])
        score_2_1 -= brightness_match_2_1
        
        if info2['bottom_edge_density'] > info1['top_edge_density']:
            score_2_1 += 100
        
        print(f"  Config 1→2 score: {score_1_2:.1f}")
        print(f"  Config 2→1 score: {score_2_1:.1f}")
        
        if score_1_2 > score_2_1:
            print(f"  ✓ Selected: Image 1 on top, Image 2 on bottom")
            return img1, img2, False
        else:
            print(f"  ✓ Selected: Image 2 on top, Image 1 on bottom")
            return img2, img1, True

    def find_horizontal_offset(self, strip1, strip2):
        """Find the best horizontal offset between two strips."""
        h1, w1 = strip1.shape[:2]
        h2, w2 = strip2.shape[:2]
        
        best_offset = 0
        best_score = float('inf')
        
        # Try different offsets
        max_offset = min(abs(w1 - w2) + 100, 200)
        
        for offset in range(-max_offset, max_offset, 2):
            # Calculate overlap region
            if offset >= 0:
                # Image 2 shifted right
                overlap_w = min(w1 - offset, w2)
                if overlap_w <= 50:
                    continue
                s1 = strip1[:, offset:offset + overlap_w]
                s2 = strip2[:, :overlap_w]
            else:
                # Image 2 shifted left
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
            
            # Calculate difference
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
        
        print(f"\n  Finding horizontal alignment...")
        
        # Use bottom strip of top image and top strip of bottom image
        strip_height = min(100, h1 // 3, h2 // 3)
        
        strip_top = img_top[-strip_height:, :]
        strip_bottom = img_bottom[:strip_height, :]
        
        # Find best horizontal offset
        offset, score = self.find_horizontal_offset(strip_top, strip_bottom)
        
        print(f"  Best horizontal offset: {offset}px")
        print(f"  Alignment quality: {255 - score:.1f}/255")
        
        # Create canvas
        canvas_w = max(w1, w2 + abs(offset))
        canvas_h = h1 + h2
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Place top image
        if offset < 0:
            # Bottom image will be to the left, so shift top image right
            x_offset_top = abs(offset)
            x_offset_bottom = 0
        else:
            # Bottom image will be to the right
            x_offset_top = 0
            x_offset_bottom = offset
        
        # Place images
        canvas[:h1, x_offset_top:x_offset_top + w1] = img_top
        canvas[h1:h1 + h2, x_offset_bottom:x_offset_bottom + w2] = img_bottom
        
        # Blend the seam
        blend_height = strip_height
        if blend_height > 0:
            seam_start = h1 - blend_height
            seam_end = h1 + blend_height
            
            if seam_end <= canvas_h:
                # Create alpha mask
                alpha = np.linspace(1, 0, blend_height * 2).reshape(-1, 1, 1)
                
                # Get overlapping regions
                overlap_x_start = max(x_offset_top, x_offset_bottom)
                overlap_x_end = min(x_offset_top + w1, x_offset_bottom + w2)
                
                if overlap_x_end > overlap_x_start:
                    # Extract regions
                    region = canvas[seam_start:seam_end, 
                                   overlap_x_start:overlap_x_end].copy()
                    
                    # Get corresponding parts from original images
                    top_region = img_top[seam_start:h1, 
                                        overlap_x_start - x_offset_top:
                                        overlap_x_end - x_offset_top]
                    bottom_region = img_bottom[:seam_end - h1,
                                              overlap_x_start - x_offset_bottom:
                                              overlap_x_end - x_offset_bottom]
                    
                    # Stack them
                    if top_region.shape[1] > 0 and bottom_region.shape[1] > 0:
                        combined = np.vstack([top_region, bottom_region])
                        
                        # Ensure same size as alpha
                        min_h = min(combined.shape[0], alpha.shape[0])
                        min_w = min(combined.shape[1], region.shape[1])
                        
                        # Blend
                        if min_h > 0 and min_w > 0:
                            blended = combined[:min_h, :min_w]
                            canvas[seam_start:seam_start + min_h,
                                  overlap_x_start:overlap_x_start + min_w] = blended
        
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
        """Main reconstruction method."""
        print(f"\n{'='*60}")
        print(f"ADVANCED IMAGE RECONSTRUCTION")
        print(f"{'='*60}\n")
        
        print(f"Loading images...")
        img1, img2 = self.load_images(image1_path, image2_path)
        print(f"  Image 1: {img1.shape}")
        print(f"  Image 2: {img2.shape}")
        
        # Determine correct order
        img_top, img_bottom, swapped = self.determine_image_order(img1, img2)
        
        # Stitch with alignment
        result, offset = self.stitch_vertical_with_offset(img_top, img_bottom)
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS! Saved to: {output_path}")
        print(f"  Output size: {result.shape}")
        print(f"  Images swapped: {swapped}")
        print(f"  Horizontal offset: {offset}px")
        print(f"{'='*60}\n")
        
        # Save to database
        self.save_to_database(
            image1_path, image2_path, output_path, 
            "success_vertical", "content_analysis", 0, offset, 0
        )
        
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
            (reconstruction_id, keypoints, matches, 0),
        )

        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def print_usage():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("ADVANCED IMAGE RECONSTRUCTION TOOL")
    print("With intelligent ordering and alignment")
    print("=" * 60)
    print("\nUsage:")
    print("  python3 v1.py <image1> <image2> [output]")
    print("\nFeatures:")
    print("  ✓ Automatic image ordering (top/bottom)")
    print("  ✓ Content-aware analysis")
    print("  ✓ Horizontal offset correction")
    print("  ✓ Prevents mirror effects")
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
            image1_path, image2_path, output_path
        )
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
