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

    def find_overlap_region(self, img1, img2, direction="vertical"):
        """Find the overlap region between two images."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        print(f"  Analyzing overlap region...")
        
        if direction == "vertical":
            # Check if img1 bottom overlaps with img2 top
            max_overlap = min(h1, h2) // 2
            best_overlap = 0
            best_score = -1
            best_offset = 0
            
            # Try different overlap heights
            for overlap_h in range(20, max_overlap, 10):
                # Get strips
                strip1 = img1[-overlap_h:, :]
                strip2 = img2[:overlap_h, :]
                
                # Try different horizontal offsets
                max_offset = abs(w1 - w2) + 50
                for offset_x in range(-max_offset, max_offset, 5):
                    # Align strips horizontally
                    if offset_x >= 0:
                        # Shift img2 right
                        common_w = min(w1, w2 - offset_x)
                        if common_w <= 0:
                            continue
                        s1 = strip1[:, :common_w]
                        s2 = strip2[:, offset_x:offset_x + common_w]
                    else:
                        # Shift img2 left
                        offset_x_abs = abs(offset_x)
                        common_w = min(w1 - offset_x_abs, w2)
                        if common_w <= 0:
                            continue
                        s1 = strip1[:, offset_x_abs:offset_x_abs + common_w]
                        s2 = strip2[:, :common_w]
                    
                    if s1.shape != s2.shape or s1.size == 0:
                        continue
                    
                    # Calculate similarity
                    diff = cv2.absdiff(s1, s2)
                    score = -np.mean(diff)  # Negative because lower diff is better
                    
                    if score > best_score:
                        best_score = score
                        best_overlap = overlap_h
                        best_offset = offset_x
            
            print(f"  Best vertical overlap: {best_overlap}px")
            print(f"  Horizontal offset: {best_offset}px")
            print(f"  Alignment score: {-best_score:.2f}")
            
            return best_overlap, best_offset, direction
            
        else:
            # Horizontal overlap (similar logic)
            max_overlap = min(w1, w2) // 2
            best_overlap = 0
            best_score = -1
            best_offset = 0
            
            for overlap_w in range(20, max_overlap, 10):
                strip1 = img1[:, -overlap_w:]
                strip2 = img2[:, :overlap_w]
                
                max_offset = abs(h1 - h2) + 50
                for offset_y in range(-max_offset, max_offset, 5):
                    if offset_y >= 0:
                        common_h = min(h1, h2 - offset_y)
                        if common_h <= 0:
                            continue
                        s1 = strip1[:common_h, :]
                        s2 = strip2[offset_y:offset_y + common_h, :]
                    else:
                        offset_y_abs = abs(offset_y)
                        common_h = min(h1 - offset_y_abs, h2)
                        if common_h <= 0:
                            continue
                        s1 = strip1[offset_y_abs:offset_y_abs + common_h, :]
                        s2 = strip2[:common_h, :]
                    
                    if s1.shape != s2.shape or s1.size == 0:
                        continue
                    
                    diff = cv2.absdiff(s1, s2)
                    score = -np.mean(diff)
                    
                    if score > best_score:
                        best_score = score
                        best_overlap = overlap_w
                        best_offset = offset_y
            
            print(f"  Best horizontal overlap: {best_overlap}px")
            print(f"  Vertical offset: {best_offset}px")
            print(f"  Alignment score: {-best_score:.2f}")
            
            return best_overlap, best_offset, direction

    def stitch_with_alignment(self, img1, img2, overlap, offset, direction="vertical"):
        """Stitch images with proper alignment and blending."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        print(f"  Stitching with alignment...")
        
        if direction == "vertical":
            # Calculate canvas size
            canvas_w = max(w1, w2 + abs(offset))
            canvas_h = h1 + h2 - overlap
            
            # Create canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Place first image
            canvas[:h1, :w1] = img1
            
            # Calculate position for second image
            if offset >= 0:
                x_offset = offset
            else:
                x_offset = 0
                # Shift first image
                canvas[:h1, :w1] = 0
                canvas[:h1, abs(offset):abs(offset) + w1] = img1
            
            y_offset = h1 - overlap
            
            # Create alpha blend in overlap region
            if overlap > 0:
                alpha = np.linspace(1, 0, overlap).reshape(-1, 1, 1)
                
                # Get overlap regions
                overlap1_region = canvas[y_offset:y_offset + overlap, 
                                        x_offset:x_offset + w2]
                overlap2_region = img2[:overlap, :]
                
                # Ensure same shape
                common_w = min(overlap1_region.shape[1], overlap2_region.shape[1])
                overlap1_region = overlap1_region[:, :common_w]
                overlap2_region = overlap2_region[:, :common_w]
                
                if overlap1_region.shape[:2] == overlap2_region.shape[:2]:
                    blended = (
                        overlap1_region * alpha + 
                        overlap2_region * (1 - alpha)
                    ).astype(np.uint8)
                    canvas[y_offset:y_offset + overlap, 
                           x_offset:x_offset + common_w] = blended
            
            # Place rest of second image
            canvas[y_offset + overlap:y_offset + h2, 
                   x_offset:x_offset + w2] = img2[overlap:, :]
            
            # Crop empty regions
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                canvas = canvas[y:y+h, x:x+w]
            
            return canvas
            
        else:
            # Horizontal stitching
            canvas_w = w1 + w2 - overlap
            canvas_h = max(h1, h2 + abs(offset))
            
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Place first image
            canvas[:h1, :w1] = img1
            
            # Calculate position for second image
            if offset >= 0:
                y_offset = offset
            else:
                y_offset = 0
                canvas[:h1, :w1] = 0
                canvas[abs(offset):abs(offset) + h1, :w1] = img1
            
            x_offset = w1 - overlap
            
            # Create alpha blend
            if overlap > 0:
                alpha = np.linspace(1, 0, overlap).reshape(1, -1, 1)
                
                overlap1_region = canvas[y_offset:y_offset + h2, 
                                        x_offset:x_offset + overlap]
                overlap2_region = img2[:, :overlap]
                
                common_h = min(overlap1_region.shape[0], overlap2_region.shape[0])
                overlap1_region = overlap1_region[:common_h, :]
                overlap2_region = overlap2_region[:common_h, :]
                
                if overlap1_region.shape[:2] == overlap2_region.shape[:2]:
                    blended = (
                        overlap1_region * alpha + 
                        overlap2_region * (1 - alpha)
                    ).astype(np.uint8)
                    canvas[y_offset:y_offset + common_h, 
                           x_offset:x_offset + overlap] = blended
            
            canvas[y_offset:y_offset + h2, 
                   x_offset + overlap:x_offset + w2] = img2[:, overlap:]
            
            # Crop empty regions
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                canvas = canvas[y:y+h, x:x+w]
            
            return canvas

    def detect_direction(self, img1, img2):
        """Detect whether images should be stitched vertically or horizontally."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Portrait images likely stitch vertically
        if h1 > w1 and h2 > w2:
            return "vertical"
        elif w1 > h1 and w2 > h2:
            return "horizontal"
        else:
            # Mixed - try both
            return "auto"

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

        # Detect direction
        direction = self.detect_direction(img1, img2)
        print(f"\nDetected direction: {direction}")
        
        if direction == "auto":
            # Try both directions
            print(f"\nTrying vertical alignment...")
            overlap_v, offset_v, _ = self.find_overlap_region(
                img1, img2, "vertical"
            )
            
            print(f"\nTrying horizontal alignment...")
            overlap_h, offset_h, _ = self.find_overlap_region(
                img1, img2, "horizontal"
            )
            
            # Use the one with better overlap
            if overlap_v > overlap_h:
                direction = "vertical"
                overlap, offset = overlap_v, offset_v
            else:
                direction = "horizontal"
                overlap, offset = overlap_h, offset_h
            
            print(f"\nSelected direction: {direction}")
        else:
            # Find overlap for detected direction
            overlap, offset, _ = self.find_overlap_region(img1, img2, direction)
        
        # Stitch images
        print(f"\nStitching images...")
        result = self.stitch_with_alignment(img1, img2, overlap, offset, direction)
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS! Saved to: {output_path}")
        print(f"  Output size: {result.shape}")
        print(f"  Direction: {direction}")
        print(f"  Overlap: {overlap}px")
        print(f"  Offset: {offset}px")
        print(f"{'='*60}\n")

        # Save to database
        self.save_to_database(
            image1_path, image2_path, output_path, 
            f"success_{direction}", "overlap_detection", 0, overlap, 0
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
    print("With intelligent overlap detection and alignment")
    print("=" * 60)
    print("\nUsage:")
    print("  python3 v1.py <image1> <image2> [output]")
    print("\nFeatures:")
    print("  ✓ Automatic overlap detection")
    print("  ✓ Horizontal/Vertical offset correction")
    print("  ✓ Smooth alpha blending")
    print("  ✓ Works with shifted camera positions")
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
