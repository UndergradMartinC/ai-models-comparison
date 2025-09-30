import time
from dino import dino
from rfdetr import rfdetr
from yolonas import yolonas

def time_function(func, name):
    """Time the execution of a function"""
    print(f"\n--- Timing {name} ---")
    start_time = time.time()
    result = func()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{name} execution time: {execution_time:.4f} seconds")
    print(f"Result: {result}")
    return execution_time

def main():
    """Main function to run and time all AI models"""
    print("AI Models Comparison - Timing Analysis")
    print("=" * 50)
    
    # Dictionary to store timing results
    timing_results = {}
    
    # Time each function
    timing_results['DINO'] = time_function(dino, 'DINO')
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

if __name__ == "__main__":
    main()