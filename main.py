# 1. Import the InferencePipeline library
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
    # Kiểm tra phím 'q' để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        pipeline.terminate()  # Dừng pipeline
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ
        cv2.waitKey(1)
    # Do something with the predictions of each frame
    print(result)


# 2. Initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="B7cKlm47FSrkbNSZ7GPl",
    workspace_name="giang-znvoi",
    workflow_id="detect-count-and-visualize",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)

# 3. Start the pipeline and wait for it to finish
pipeline.start()
pipeline.join()