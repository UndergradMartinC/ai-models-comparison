import time
#from dino import dino
from rfdetr import rfdetr
#from yolox import yolox
import os

def test_rfdetr(photo_name, use_gpu=False):
    """Test the model across all photos in the test_photos directory"""
    start_time = time.time()
    rfdetr(f'test_photos/images/{photo_name}', f'test_photos/labels/{photo_name.replace(".jpeg", ".json")}', use_gpu=use_gpu)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"RF-DETR execution time: {execution_time:.4f} seconds")

def main():
    """Main function to run and time all AI models
    print("AI Models Comparison - Timing Analysis")
    print("=" * 50)
    
    # Dictionary to store timing results
    timing_results = {}
    
    # Time each function
    timing_results['RF-DETR'] = time_function(rfdetr, 'RF-DETR')
    
    # Print summary
    print("\n" + "=" * 50)
    print("TIMING SUMMARY")
    print("=" * 50)
    
    # Sort by execution time
    sorted_results = sorted(timing_results.items(), key=lambda x: x[1])
    
    for i, (model, time_taken) in enumerate(sorted_results, 1):
        print(f"{i}. {model:10s}: {time_taken:.4f} seconds")
    
    print(f"\nFastest: {sorted_results[0][0]} ({sorted_results[0][1]:.4f}s)")
    print(f"Slowest: {sorted_results[-1][0]} ({sorted_results[-1][1]:.4f}s)")
    """

    test_rfdetr("dense1.jpeg", use_gpu=False)

if __name__ == "__main__":
    main()