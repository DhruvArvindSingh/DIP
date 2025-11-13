import cv2
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import base64


class ImageReconstructor:
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
                method TEXT
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
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")

        return img1, img2

    def detect_features(self, img1, img2):
        """Detect features using SIFT or ORB."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Try SIFT first (better quality but may not be available)
        try:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            method = "SIFT"
        except cv2.error:
            # Fall back to ORB if SIFT not available
            orb = cv2.ORB_create(nfeatures=5000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            method = "ORB"

        return kp1, des1, kp2, des2, method

    def match_features(self, des1, des2, method):
        """Match features between two images."""
        if method == "SIFT":
            # FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher for ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        matches = matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches

    def stitch_horizontal(self, img1, img2, kp1, kp2, matches):
        """Stitch images horizontally based on matched features."""
        if len(matches) < 4:
            raise ValueError(
                f"Not enough matches found: {len(matches)}. "
                "Images may not be related."
            )

        # Extract matched keypoints
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to img2's perspective
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Calculate output size
        corners1 = np.float32(
            [[0, 0], [0, height1], [width1, height1], [width1, 0]]
        ).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)

        corners2 = np.float32(
            [[0, 0], [0, height2], [width2, height2], [width2, 0]]
        ).reshape(-1, 1, 2)

        all_corners = np.concatenate(
            (corners1_transformed, corners2), axis=0
        )

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation = np.array(
            [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]
        )

        # Warp and combine
        result = cv2.warpPerspective(
            img1, translation.dot(H), (x_max - x_min, y_max - y_min)
        )
        result[-y_min : height2 - y_min, -x_min : width2 - x_min] = img2

        return result, len(matches)

    def simple_concatenate(self, img1, img2, direction="horizontal"):
        """Simple concatenation if feature matching fails."""
        if direction == "horizontal":
            # Resize to same height
            height = max(img1.shape[0], img2.shape[0])
            img1_resized = cv2.resize(img1, (img1.shape[1], height))
            img2_resized = cv2.resize(img2, (img2.shape[1], height))
            result = np.hstack((img1_resized, img2_resized))
        else:  # vertical
            # Resize to same width
            width = max(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (width, img1.shape[0]))
            img2_resized = cv2.resize(img2, (width, img2.shape[0]))
            result = np.vstack((img1_resized, img2_resized))

        return result

    def reconstruct(
        self, image1_path, image2_path, output_path="reconstructed.jpg"
    ):
        """Main reconstruction method."""
        print(f"Loading images...")
        img1, img2 = self.load_images(image1_path, image2_path)

        print(f"Detecting features...")
        kp1, des1, kp2, des2, method = self.detect_features(img1, img2)
        print(f"Method: {method}")
        print(f"Keypoints - Image1: {len(kp1)}, Image2: {len(kp2)}")

        print(f"Matching features...")
        matches = self.match_features(des1, des2, method)
        print(f"Good matches found: {len(matches)}")

        try:
            if len(matches) >= 10:
                print("Stitching with feature matching...")
                result, match_count = self.stitch_horizontal(
                    img1, img2, kp1, kp2, matches
                )
                status = "success_feature_matching"
            else:
                print(
                    "Not enough matches. Using simple concatenation..."
                )
                result = self.simple_concatenate(img1, img2)
                match_count = len(matches)
                status = "success_concatenation"

            # Save result
            cv2.imwrite(output_path, result)
            print(f"Reconstructed image saved to: {output_path}")

            # Save to database
            self.save_to_database(
                image1_path,
                image2_path,
                output_path,
                status,
                method,
                len(kp1) + len(kp2),
                match_count,
            )

            return result

        except Exception as e:
            print(f"Error during reconstruction: {e}")
            status = "failed"
            self.save_to_database(
                image1_path, image2_path, output_path, status, method, 0, 0
            )
            raise

    def save_to_database(
        self,
        img1_path,
        img2_path,
        output_path,
        status,
        method,
        keypoints,
        matches,
    ):
        """Save reconstruction information to database."""
        self.cursor.execute(
            """
            INSERT INTO reconstructions
            (image1_path, image2_path, output_path, status, method)
            VALUES (?, ?, ?, ?, ?)
        """,
            (img1_path, img2_path, output_path, status, method),
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
                   r.created_at, r.status, r.method,
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


# Example usage
if __name__ == "__main__":
    reconstructor = ImageReconstructor()

    # Example: Reconstruct two torn images
    try:
        image1_path = "torn_image_1.jpg"
        image2_path = "torn_image_2.jpg"
        output_path = "reconstructed_image.jpg"

        result = reconstructor.reconstruct(
            image1_path, image2_path, output_path
        )

        print("\n" + "=" * 50)
        print("Recent reconstruction history:")
        print("=" * 50)
        history = reconstructor.get_history(5)
        for record in history:
            print(f"\nID: {record[0]}")
            print(f"Status: {record[5]}")
            print(f"Method: {record[6]}")
            print(f"Matches: {record[7]}")
            print(f"Created: {record[4]}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease provide valid image paths.")
    finally:
        reconstructor.close()
