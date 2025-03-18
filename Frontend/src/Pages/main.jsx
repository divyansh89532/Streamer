import { useState, useEffect, useRef } from "react";
import { 
  FaRegSmileWink, FaHeadSideMask, FaBrain, FaSearch,
  FaChevronLeft, FaChevronRight, FaChevronDown, FaSun, FaMoon
} from "react-icons/fa";
import { GiHairStrands } from "react-icons/gi";
import { FaHelmetSafety } from "react-icons/fa6";
import { motion, AnimatePresence } from "framer-motion";
import { ClipLoader } from "react-spinners";

// Hardcoded descriptions for each model
const modelDescriptions = {
  // Face: "This is a lightweight, real-time optimized solution for facial recognition. Designed for efficiency, it ensures high-speed processing without compromising accuracy. Its streamlined architecture makes it ideal for applications requiring quick response times, such as surveillance, authentication, and attendance tracking. With minimal computational overhead, Face integrates seamlessly into edge devices, mobile applications, and cloud-based systems. Whether for security, AI-driven interactions, or smart monitoring, it delivers reliable performance with low latency.",
  Mask: "This is a lightweight, real-time optimized solution for mask detection. Designed for efficiency, it ensures high-speed processing without compromising accuracy. Its streamlined architecture makes it ideal for applications requiring quick response times, such as compliance monitoring, workplace safety, and public health enforcement. With minimal computational overhead, Mask integrates seamlessly into edge devices, mobile applications, and cloud-based systems. Whether for security screenings, or AI-driven safety measures, it delivers reliable performance with low latency.",
  Helmet: "This is a lightweight, real-time optimized solution for safety helmet detection. Designed for efficiency, it ensures high-speed processing without compromising accuracy. Its streamlined architecture makes it ideal for applications requiring quick response times, such as construction site monitoring, industrial safety compliance, and workplace security. With minimal computational overhead, Helmet integrates seamlessly into edge devices, mobile applications, or cloud-based systems. Whether for worker protection, AI-driven safety enforcement, or automated monitoring, it delivers reliable performance with low latency.",
  Hairnet: "This is a lightweight, real-time optimized solution for hairnet detection. Designed for efficiency, it ensures high-speed processing without compromising accuracy. Its streamlined architecture makes it ideal for applications requiring quick response times, such as food processing, healthcare, and hygiene compliance. With minimal computational overhead, Hairnet integrates seamlessly into edge devices, mobile applications, and cloud-based systems. Whether for quality control, or AI-driven safety monitoring, it delivers reliable performance with low latency.",
  Face: "This is a user-trained custom model for specific object detection.",
};

// Icons for each model
const modelIcons = {
  // Face: <FaRegSmileWink className="text-blue-500 text-2xl" />,
  Mask: <FaHeadSideMask className="text-green-500 text-2xl" />,
  Helmet: <FaHelmetSafety className="text-orange-500 text-2xl" />,
  Hairnet: <GiHairStrands className="text-purple-500 text-2xl" />,
  Face: <FaRegSmileWink className="text-blue-500 text-2xl" />,
};

const ImageCarousel = ({ model, backendURI }) => {
  const [images, setImages] = useState([]);
  const [loadingImages, setLoadingImages] = useState(false);
  const [index, setIndex] = useState(0);
  const imagesPerSlide = window.innerWidth < 768 ? 2 : 3;

  useEffect(() => {
    if (!model) return;
    setLoadingImages(true);

    const fetchImages = () => {
      fetch(`${backendURI}/images/${model}`)
        .then((res) => res.json())
        .then((data) => {
          const fetchedImages = data.map((record) => record.image_url);
          const uniqueImages = Array.from(new Set(fetchedImages));
          setImages(uniqueImages.slice(-10));
          setLoadingImages(false);
          setIndex(0);
        })
        .catch((error) => {
          console.error("Error fetching images:", error);
          setLoadingImages(false);
        });
    };

    fetchImages();
    const intervalId = setInterval(fetchImages, 3000);
    return () => clearInterval(intervalId);
  }, [model, backendURI]);

  useEffect(() => {
    if (images.length > 0) {
      const interval = setInterval(() => {
        setIndex((prevIndex) => (prevIndex + imagesPerSlide) % images.length);
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [images, imagesPerSlide]);

  if (loadingImages) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <ClipLoader color="#000" size={50} />
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <p>No images found</p>
      </div>
    );
  }

  const displayedImages =
  images.length >= imagesPerSlide
    ? Array.from({ length: imagesPerSlide }, (_, i) => images[(index + i) % images.length])
    : images;


  return (
    <div className="w-full h-full overflow-hidden relative rounded-lg shadow-lg bg-gray-100 dark:bg-gray-700 p-1 flex items-center justify-center">
      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          className="flex flex-row md:flex-col gap-2 items-center justify-center"
          initial={{ x: 200, opacity: 0, scale: 0.8 }}
          animate={{ x: 0, opacity: 1, scale: 1 }}
          exit={{ x: 200, opacity: 0, scale: 0.8 }}
          transition={{ duration: 1, ease: "easeInOut" }}
        >
          {displayedImages.map((img, i) => (
            <motion.img
              key={i}
              src={img}
              className="w-[200px] h-[180px] object-cover rounded-lg shadow-md"
              whileHover={{ scale: 1.1 }}
              style={{
                // marginTop: i === 0 && window.innerWidth >= 768 ? "2.5rem" : 0,
                marginLeft: i === 0 && window.innerWidth < 768 ? "2.2rem" : 0,
              }}
            />
          ))}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

const VideoPlaceholder = ({ streamUrl }) => {
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!streamUrl) return;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    const newWs = new WebSocket(streamUrl);
    wsRef.current = newWs;
    setLoading(true);
    newWs.onmessage = (event) => {
      setLoading(false);
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      if (videoRef.current) {
        videoRef.current.src = url;
      }
    };
    newWs.onclose = () => {
      console.log("WebSocket closed");
      setLoading(false);
    };
    newWs.onerror = (error) => {
      console.error("WebSocket Error:", error);
      setLoading(false);
    };
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [streamUrl]);

  return (
    <div className="w-full relative flex items-center justify-center overflow-hidden max-h-[600px] bg-gray-200 dark:bg-gray-800">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50 rounded-lg">
          <ClipLoader color="#fff" size={50} />
        </div>
      )}
      {streamUrl ? (
        <img
          ref={videoRef}
          src="https://dummyimage.com/800x550/000/fff"
          alt="Live Stream"
          className="w-full h-auto rounded-lg shadow-lg max-h-[750px]"
        />
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col items-center justify-center w-full h-[300px] bg-gray-200 dark:bg-gray-800 rounded-lg shadow-md p-6 text-center"
        >
          <FaSearch className="text-gray-500 dark:text-gray-300 text-6xl mb-4 animate-bounce" />
          <h2 className="text-xl font-semibold text-gray-700 dark:text-gray-200">No Model Selected</h2>
          <p className="text-gray-500 dark:text-gray-400 mt-2">Choose a model to start live detection.</p>
        </motion.div>
      )}
    </div>
  );
};

const DropdownButton = ({ selectedModel, models, onSelect }) => {
  const [isHovered, setIsHovered] = useState(false);
  const timeoutRef = useRef(null);

  const handleMouseEnter = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setIsHovered(false);
    }, 300);
  };

  return (
    <div 
      className="relative"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="flex items-center justify-between gap-2 w-full px-4 py-2 rounded-lg shadow transition-all bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 hover:bg-blue-50 dark:hover:bg-blue-900 hover:shadow-md text-sm text-gray-800 dark:text-gray-200"
      >
        {selectedModel ? (
          <>
            <div className="flex items-center gap-2">
              {modelIcons[selectedModel]}
              <span>{selectedModel}</span>
            </div>
            <FaChevronDown className="text-gray-500 dark:text-gray-300" />
          </>
        ) : (
          <>
            <span>Select a Model</span>
            <FaChevronDown className="text-gray-500 dark:text-gray-300" />
          </>
        )}
      </motion.button>

      {isHovered && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="absolute z-10 mt-2 w-48 rounded-lg shadow-lg bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600"
        >
          <div className="py-1">
            {models.map((model) => (
              <motion.button
                key={model}
                onClick={() => {
                  onSelect(model);
                  setIsHovered(false);
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`flex items-center gap-2 px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-blue-50 dark:hover:bg-blue-900 w-full text-left ${
                  selectedModel === model ? "bg-blue-100 dark:bg-blue-900" : ""
                }`}
              >
                {modelIcons[model]} <span>{model}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

const AvailableModels = ({ models, onSelect, selectedModel }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (selectedModel && containerRef.current && window.innerWidth < 768) {
      const selectedButton = containerRef.current.querySelector(
        `[data-model="${selectedModel}"]`
      );
      if (selectedButton) {
        selectedButton.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      }
    }
  }, [selectedModel]);

  const handleScroll = (direction) => {
    if (containerRef.current) {
      const scrollAmount = 200;
      if (direction === "left") {
        containerRef.current.scrollBy({ left: -scrollAmount, behavior: "smooth" });
      } else if (direction === "right") {
        containerRef.current.scrollBy({ left: scrollAmount, behavior: "smooth" });
      }
    }
  };

  return (
    <div className="mb-4">
      <div className="relative hidden md:block">
        <DropdownButton 
          selectedModel={selectedModel} 
          models={models} 
          onSelect={onSelect} 
        />
      </div>
      <div className="md:hidden relative">
        <button
          onClick={() => handleScroll("left")}
          className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1/2 bg-white dark:bg-gray-700 bg-opacity-75 rounded-full p-2 shadow-md z-10"
        >
          <FaChevronLeft className="h-6 w-6 text-gray-700 dark:text-gray-300" />
        </button>

        <button
          onClick={() => handleScroll("right")}
          className="absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-1/2 bg-white dark:bg-gray-700 bg-opacity-75 rounded-full p-2 shadow-md z-10"
        >
          <FaChevronRight className="h-6 w-6 text-gray-700 dark:text-gray-300" />
        </button>

        <div
          ref={containerRef}
          className="flex items-center overflow-x-auto gap-2 py-2"
          style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
        >
          {models.map((model) => (
            <motion.button
              key={model}
              data-model={model}
              onClick={() => onSelect(model)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`flex-shrink-0 flex items-center gap-2 px-4 py-2 rounded-lg shadow transition-all bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 hover:bg-blue-50 dark:hover:bg-blue-900 text-sm ${
                selectedModel === model ? "bg-blue-100 dark:bg-blue-900 border-blue-400" : ""
              }`}
            >
              {modelIcons[model]} <span className="text-gray-800 dark:text-gray-200">{model}</span>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
};

const ModelDescription = ({ model }) => (
  <motion.div
    key={model}
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.3 }}
    className="mb-4 p-4 rounded-lg bg-transparent border border-gray-200 dark:border-gray-600 shadow-sm"
  >
    <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
      {modelIcons[model]} <span>Model Details</span>
    </h3>
    <p className="text-gray-600 dark:text-gray-300 mt-2">{modelDescriptions[model] || "Select a model to see details."}</p>
  </motion.div>
);

const YoloModelComponent = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  // const modelsList = ["Face", "Mask", "Helmet", "Hairnet", "Custom"]; 
  const modelsList = ["Mask", "Helmet", "Hairnet", "Face"]; 
  const backendURI = "http://localhost:8000";

  const modelEndpoints = {
    // Face: `${backendURI}/ws/face`,
    Mask: `${backendURI}/ws/mask`,
    Helmet: `${backendURI}/ws/helmet`,
    Hairnet: `${backendURI}/ws/hairnet`,
    Face: `${backendURI}/ws/verify`,
  };

  return (
    <div className={`${darkMode ? "dark" : ""}`}>

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col md:flex-row p-3 gap-6 bg-gray-100 dark:bg-gray-800 min-h-screen overflow-hidden"
      >
        {/* Left Column: Video */}
        <div className="flex-grow flex items-center justify-center w-full">
          <VideoPlaceholder
            streamUrl={selectedModel ? modelEndpoints[selectedModel] : null}
          />
        </div>

        {/* Middle Column: Carousel */}
        {selectedModel && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
            className="w-full md:w-2/8 h-auto flex justify-center items-center mt-4 md:mt-0"
          >
            <ImageCarousel model={selectedModel} backendURI={backendURI} />
          </motion.div>
        )}
      </motion.div>

      {/* Overlay Sidebar (opens from the right) */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            key="sidebar"
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
            className="absolute top-0 right-0 w-full md:w-4/12 p-5 z-50 backdrop-blur-md"
          >
            <div className="w-full p-5 border border-gray-200 dark:border-gray-600 rounded-lg bg-transparent dark:bg-gray-800">
              <div className="mb-6 text-center">
                <h2 className="text-2xl font-bold text-blue-600">Model Selection</h2>
                <p className="text-sm text-gray-600 dark:text-gray-300">Please select a model to start live detection.</p>
              </div>
              <AvailableModels
                models={modelsList}
                onSelect={setSelectedModel}
                selectedModel={selectedModel}
              />
              {selectedModel ? (
                <>
                  <hr className="my-4 border-t border-gray-300 dark:border-gray-600" />
                  <ModelDescription model={selectedModel} />
                </>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                  className="p-4 mt-4 text-center text-gray-500 dark:text-gray-300 italic"
                >
                  Waiting for model selection...
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button for Sidebar */}
      <button
        onClick={() => setSidebarOpen((prev) => !prev)}
        className="absolute right-0 top-0 m-4 bg-blue-600 text-white p-2 rounded-full shadow-lg z-50"
      >
        {sidebarOpen ? <FaChevronRight /> : <FaChevronLeft />}
      </button>
    </div>
  );
};

export default YoloModelComponent;
