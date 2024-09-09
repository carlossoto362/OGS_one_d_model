LINE := export OGS_ONE_D_HOME_PATH=$(PWD)
BASHRC := $(HOME)/.bashrc


setup: requirements.txt add_home_path
	python3 -m venv OGS_env
	./OGS_env/bin/pip install networkx==2.8.8
	./OGS_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	./OGS_env/bin/pip install -r requirements.txt
	cp -r ./PySurfaceData ./OGS_env/lib/python3.8/site-packages/
	@echo run 'source ./OGS_env/bin/activate' for activating OGS env, 'deactivate' for deactivating it. 

add_home_path:
	@grep -qxF $(LINE) $(BASHRC) || echo $(LINE) >> $(BASHRC)

clean:
	@rm -f -r ./OGS_env
	@sed -i "\|$(LINE)|d" $(BASHRC)

