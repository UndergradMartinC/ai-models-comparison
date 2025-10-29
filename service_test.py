import os
import requests
import time
from pathlib import Path

# Change this to your API endpoint
IP = "https://latest-160413876167.us-central1.run.app"  # Cloud endpoint
# IP = "http://localhost:8080"  # For local testing

def test_photo(photo_path, json_path, use_gpu=True, create_overlay=True):
    """Test a single photo with its corresponding reference JSON."""
    url = f"{IP}/run"
    
    with open(photo_path, "rb") as img_file, open(json_path, "rb") as json_file:
        files = {
            "image": (os.path.basename(photo_path), img_file, "image/jpeg"),
            "reference_json": (os.path.basename(json_path), json_file, "application/json"),
        }
        data = {
            "use_gpu": str(use_gpu).lower(),
            "create_overlay": str(create_overlay).lower(),
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, files=files, data=data, timeout=300)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                return response.json(), elapsed_time
            else:
                return {"error": response.text, "status_code": response.status_code}, elapsed_time
        except requests.exceptions.ConnectionError as e:
            elapsed_time = time.time() - start_time
            return {"error": f"Connection failed: {str(e)}", "status_code": None, "connection_error": True}, elapsed_time
        except requests.exceptions.Timeout as e:
            elapsed_time = time.time() - start_time
            return {"error": f"Request timed out: {str(e)}", "status_code": None}, elapsed_time
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {"error": f"Unexpected error: {str(e)}", "status_code": None}, elapsed_time


def main():
    images_dir = Path("test_photos/images")
    labels_dir = Path("test_photos/labels")
    
    # Get all image files
    image_files = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_files:
        print("No images found in test_photos/images")
        return
    
    print(f"🌐 Testing endpoint: {IP}")
    print(f"📁 Found {len(image_files)} images to test")
    print("=" * 80)
    
    # Cold start test with first image
    first_image = image_files[0]
    first_json = labels_dir / f"{first_image.stem}.json"
    
    if not first_json.exists():
        print(f"Warning: No JSON found for {first_image.name}, skipping cold start")
    else:
        print(f"\n🧪 COLD START TEST: {first_image.name}")
        print("-" * 80)
        result, elapsed = test_photo(str(first_image), str(first_json))
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            if result.get('status_code'):
                print(f"Status code: {result.get('status_code')}")
            if result.get('connection_error'):
                print(f"\n💡 Tip: Make sure the API server is running.")
                print(f"   - For local: Start the server with 'python dinoAPI.py'")
                print(f"   - For cloud: Check if the endpoint URL is correct")
            return
        else:
            print(f"✅ Success")
            print(f"⏱️  Request time: {elapsed:.2f}s")
            exec_time = result.get('execution_time_seconds', 0)
            if exec_time:
                print(f"⏱️  Server execution time: {exec_time:.2f}s")
            
            # Print metrics if available
            comp_results = result.get("comparison_results", {})
            if comp_results:
                print(f"\n📊 Metrics:")
                print(f"   Mean AP: {comp_results.get('mean_ap', 'N/A'):.3f}")
                print(f"   Mean F1: {comp_results.get('mean_f1', 'N/A'):.3f}")
                print(f"   Mean Accuracy: {comp_results.get('mean_accuracy', 'N/A'):.3f}")
        
        print("=" * 80)
    
    # Loop through remaining images
    remaining_images = image_files[1:] if len(image_files) > 1 else []
    
    if remaining_images:
        print(f"\n🔄 Testing {len(remaining_images)} remaining images...")
        print("=" * 80)
        
        results_summary = []
        
        for idx, image_path in enumerate(remaining_images, start=2):
            json_path = labels_dir / f"{image_path.stem}.json"
            
            if not json_path.exists():
                print(f"\n[{idx}/{len(image_files)}] ⚠️  Skipping {image_path.name} - no matching JSON")
                continue
            
            print(f"\n[{idx}/{len(image_files)}] Testing: {image_path.name}")
            result, elapsed = test_photo(str(image_path), str(json_path))
            
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
                results_summary.append({
                    "image": image_path.name,
                    "status": "error",
                    "elapsed": elapsed,
                })
            else:
                comp_results = result.get("comparison_results", {})
                mean_ap = comp_results.get('mean_ap', 0)
                mean_f1 = comp_results.get('mean_f1', 0)
                mean_acc = comp_results.get('mean_accuracy', 0)
                
                print(f"   ✅ AP: {mean_ap:.3f} | F1: {mean_f1:.3f} | Acc: {mean_acc:.3f} | Time: {elapsed:.2f}s")
                
                results_summary.append({
                    "image": image_path.name,
                    "status": "success",
                    "elapsed": elapsed,
                    "mean_ap": mean_ap,
                    "mean_f1": mean_f1,
                    "mean_accuracy": mean_acc,
                })
        
        # Summary
        print("\n" + "=" * 80)
        print("📈 SUMMARY")
        print("=" * 80)
        
        successful = [r for r in results_summary if r["status"] == "success"]
        failed = [r for r in results_summary if r["status"] == "error"]
        
        print(f"Total tests: {len(results_summary)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_ap = sum(r["mean_ap"] for r in successful) / len(successful)
            avg_f1 = sum(r["mean_f1"] for r in successful) / len(successful)
            avg_acc = sum(r["mean_accuracy"] for r in successful) / len(successful)
            avg_time = sum(r["elapsed"] for r in successful) / len(successful)
            
            print(f"\n📊 Average Metrics (successful tests):")
            print(f"   Mean AP: {avg_ap:.3f}")
            print(f"   Mean F1: {avg_f1:.3f}")
            print(f"   Mean Accuracy: {avg_acc:.3f}")
            print(f"   Avg Time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()