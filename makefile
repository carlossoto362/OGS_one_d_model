ifeq ($(shell uname), Darwin)
        SHRC := $(HOME)/.zshrc
	SED_INPLACE := sed -i ''
else
        SHRC := $(HOME)/.bashrc
	SED_INPLACE := sed -i
endif

LINE := export OGS_ONE_D_HOME_PATH=$(PWD)
PYTHON_VERSION := $(shell python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")

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
	grep -qxF '$(LINE)' $(SHRC) || echo $(LINE) >> $(SHRC)
	grep -qxF 'export OGS_env_path=$(ENVDIR)' $(SHRC) || echo export OGS_env_path=$(ENVDIR) >> $(SHRC)
	make setup ENVDIR=$(ENVDIR)

setup: requirements.txt
	echo $(ENVDIR)
	$(ENVDIR)/OGS_env/bin/pip install networkx==2.8.8
	$(ENVDIR)/OGS_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(ENVDIR)/OGS_env/bin/pip install -r requirements.txt
	cp -r ./PySurfaceData $(ENVDIR)/OGS_env/lib/$(PYTHON_VERSION)/site-packages/
	@echo run 'source $(ENVDIR)/OGS_env/bin/activate' for activating OGS env, 'deactivate' for deactivating it. 
	@echo also run 'source $(SHRC)' to let know to the scripts where is the home directory of OGS. 

clean:
	@rm -f -r $(OGS_env_path)/OGS_env
	@$(SED_INPLACE) "\|$(LINE)|d" $(SHRC)
	@$(SED_INPLACE) "\|export OGS_env_path=$(ENVDIR)|d" $(SHRC)

