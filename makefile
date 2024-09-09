LINE := export OGS_ONE_D_HOME_PATH=$(PWD)
BASHRC := $(HOME)/.bashrc

create_venv:
	@read -p "Enter the directory where you want to create the virtual environment (empty for current one): " ENVDIR; \
	if [ -z "$$ENVDIR" ]; then \
		ENVDIR="."; \
		echo "Using the current directory"; \
	else \
		mkdir -p "$$ENVDIR"; \
		python3 -m venv "$$ENVDIR/OGS_env"; \
		echo "Virtual environment created in $$ENVDIR/OGS_env"; \
	fi; \
	make add_bashrc_line ENVDIR=$$ENVDIR

add_bashrc_line:
	grep -qxF $(LINE) $(BASHRC) || echo $(LINE) >> $(BASHRC)
	grep -qxF export OGS_env_path=$(ENVDIR) $(BASHRC) || echo export OGS_env_path=$(ENVDIR) >> $(BASHRC)
	make setup ENVDIR=$(ENVDIR)

setup: requirements.txt
	echo $(ENVDIR)
	$(ENVDIR)/OGS_env/bin/pip install networkx==2.8.8
	$(ENVDIR)/OGS_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(ENVDIR)/OGS_env/bin/pip install -r requirements.txt
	cp -r ./PySurfaceData $(ENVDIR)/OGS_env/lib/python3.8/site-packages/
	@echo run 'source $(ENVDIR)/OGS_env/bin/activate' for activating OGS env, 'deactivate' for deactivating it. 

clean:
	@rm -f -r $(ENVDIR)/OGS_env
	@sed -i "\|$(LINE)|d" $(BASHRC)
	@sed -i "\|export OGS_env_path=$(ENVDIR)|d" $(BASHRC)

