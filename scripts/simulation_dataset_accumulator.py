import zarr
import os

class SimulationDatasetAccumulator:
    def __init__(self, merged_path):
        """
        merged_path:
            長期累積用的 simulation_dataset_merged.zarr.zip
        """
        self.merged_path = merged_path

    def _ask_user(self):
        print(f"\n⚠️ Merged dataset already exists:")
        print(f"   {self.merged_path}")
        print("Choose an action:")
        print("  yes / y  → 合併新 dataset（append episodes）")
        print("  no / n   → 覆蓋 merged dataset")
        print("  cancel / c   → 取消本次更新")

        while True:
            ans = input("Your choice [yes/no/cancel]: ").strip().lower()
            if ans in ["yes", "y"]:
                return "merge"
            if ans in ["no", "n"]:
                return "overwrite"
            if ans in ["cancel", "c"]:
                return "cancel"

    def _open_merged(self):
        if os.path.exists(self.merged_path):
            action = self._ask_user()
            if action == "cancel":
                print("❌ Update cancelled.")
                return None
            if action == "overwrite":
                print("🧹 Removing existing merged dataset...")
                os.remove(self.merged_path)
        return zarr.open(self.merged_path, mode="a")

    def update_with(self, new_dataset_path):
        """
        new_dataset_path:
            剛產生的 simulation_dataset.zarr.zip
        """
        print(f"\n📥 New dataset: {new_dataset_path}")

        root_new = zarr.open(new_dataset_path, mode="r")
        root_merged = self._open_merged()
        if root_merged is None:
            return

        episode_counter = len(root_merged.keys())
        print(f"📦 Starting from episode index = {episode_counter}")

        for ep_name in root_new.keys():
            ep_new = root_new[ep_name]
            ep_out_name = f"episode_{episode_counter:05d}"
            ep_out = root_merged.create_group(ep_out_name)

            for key in ep_new.keys():
                ep_out.create_dataset(
                    key,
                    data=ep_new[key][:],
                    chunks=True,
                    compressor=zarr.Blosc(
                        cname="zstd",
                        clevel=5,
                        shuffle=2,
                    ),
                )

            print(f"  ✔ Appended {ep_out_name}")
            episode_counter += 1

        print(f"\n✅ Update finished. Total episodes = {episode_counter}")
