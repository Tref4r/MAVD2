import requests
import os

def send_video(video_path: str, api_url: str = "http://mmaction:8000/process") -> None:
    """
    Send a video file to server and receive the result
    
    Args:
        video_path (str): Path to the video file
        api_url (str): URL of the API endpoint
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist!")
        return
        
    print(f"Processing video: {video_path}")
    
    try:
        # Open and send video file
        with open(video_path, 'rb') as video_file:
            files = {
                'file': (os.path.basename(video_path), video_file, 'video/mp4')
            }
            
            # Send POST request
            response = requests.post(api_url, files=files)
            
            # Handle response
            if response.status_code == 200:
                # Create output filename
                output_name = os.path.basename(video_path).replace('.mp4', '__pose.npy')
                output_dir = 'pose_feats'
                output_path = os.path.join(output_dir, output_name)
                
                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save result file
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Result saved to: {output_path}")
                return output_path
            else:
                print(f"Error: {response.status_code}")
                print(f"Details: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Connection error to server!")
    except Exception as e:
        print(f"Error: {str(e)}")
