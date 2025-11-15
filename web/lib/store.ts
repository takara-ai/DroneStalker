import { create } from "zustand";

type UnlockStore = {
  unlockedMotion: boolean;
  unlockedFire: boolean;
  unlockedTracking: boolean;
  unlockedAutoFire: boolean;
  unlockedMotionPrediction: boolean;
  unlockedCode: boolean;
  unlockedCredits: boolean;
  tab: "video" | "code" | "credits";
  activeCamera: boolean;
  activeMotionDetection: boolean;
  activeFire: boolean;
  activeTracking: boolean;
  activeAutoFire: boolean;
  activeMotionPrediction: boolean;
  setUnlockedMotion: (value: boolean) => void;
  setUnlockedFire: (value: boolean) => void;
  setUnlockedTracking: (value: boolean) => void;
  setUnlockedAutoFire: (value: boolean) => void;
  setUnlockedMotionPrediction: (value: boolean) => void;
  setUnlockedCode: (value: boolean) => void;
  setUnlockedCredits: (value: boolean) => void;
  setTab: (value: "video" | "code" | "credits") => void;
  setActiveCamera: (value: boolean) => void;
  setActiveMotionDetection: (value: boolean) => void;
  setActiveFire: (value: boolean) => void;
  setActiveTracking: (value: boolean) => void;
  setActiveAutoFire: (value: boolean) => void;
  setActiveMotionPrediction: (value: boolean) => void;
};

export const useStore = create<UnlockStore>((set) => ({
  unlockedMotion: false,
  unlockedFire: false,
  unlockedTracking: false,
  unlockedAutoFire: false,
  unlockedMotionPrediction: false,
  unlockedCode: false,
  unlockedCredits: false,
  tab: "video",
  activeCamera: true,
  activeMotionDetection: false,
  activeFire: false,
  activeTracking: false,
  activeAutoFire: false,
  activeMotionPrediction: false,
  setUnlockedMotion: (value) => set({ unlockedMotion: value }),
  setUnlockedFire: (value) => set({ unlockedFire: value }),
  setUnlockedTracking: (value) => set({ unlockedTracking: value }),
  setUnlockedAutoFire: (value) => set({ unlockedAutoFire: value }),
  setUnlockedMotionPrediction: (value) =>
    set({ unlockedMotionPrediction: value }),
  setUnlockedCode: (value) => set({ unlockedCode: value }),
  setUnlockedCredits: (value) => set({ unlockedCredits: value }),
  setTab: (value) => set({ tab: value }),
  setActiveCamera: (value) => set({ activeCamera: value }),
  setActiveMotionDetection: (value) => set({ activeMotionDetection: value }),
  setActiveFire: (value) => set({ activeFire: value }),
  setActiveTracking: (value) => set({ activeTracking: value }),
  setActiveAutoFire: (value) => set({ activeAutoFire: value }),
  setActiveMotionPrediction: (value) => set({ activeMotionPrediction: value }),
}));
