import ifcopenshell
import pandas as pd
import tempfile
import os
import io
import uuid
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IFCProcessor:
    def __init__(self, ifc_file):
        if isinstance(ifc_file, (str, bytes, os.PathLike)):
            self.ifc_file = ifcopenshell.open(ifc_file)
        elif isinstance(ifc_file, io.BytesIO):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ifc')
            temp_file.write(ifc_file.getvalue())
            temp_file.close()
            self.ifc_file = ifcopenshell.open(temp_file.name)
            os.unlink(temp_file.name)
        elif hasattr(ifc_file, 'read'):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ifc')
            temp_file.write(ifc_file.read())
            temp_file.close()
            self.ifc_file = ifcopenshell.open(temp_file.name)
            os.unlink(temp_file.name)
        else:
            raise ValueError("Unsupported file type. Please provide a valid IFC file.")
        self.properties_df = None
        self.debug_logs = []

    def extract_properties(self):
        data = []
        for element in self.ifc_file.by_type("IfcProduct"):
            element_data = {"Type": element.is_a(), "Name": element.Name, "GUID": element.GlobalId}
            
            location = self.get_object_placement(element)
            if location:
                element_data.update(location)
            
            for definition in element.IsDefinedBy:
                if definition.is_a("IfcRelDefinesByProperties"):
                    property_set = definition.RelatingPropertyDefinition
                    if property_set.is_a("IfcPropertySet"):
                        for property in property_set.HasProperties:
                            if property.is_a("IfcPropertySingleValue"):
                                element_data[f"{property_set.Name}.{property.Name}"] = property.NominalValue.wrappedValue if property.NominalValue else None
                    elif property_set.is_a("IfcElementQuantity"):
                        for quantity in property_set.Quantities:
                            if quantity.is_a() in ["IfcQuantityLength", "IfcQuantityArea", "IfcQuantityVolume", "IfcQuantityCount"]:
                                value_attribute = f"{quantity.is_a()[10:]}Value"
                                if hasattr(quantity, value_attribute):
                                    element_data[f"{property_set.Name}.{quantity.Name}"] = getattr(quantity, value_attribute)
                                else:
                                    print(f"Warning: {quantity.is_a()} does not have attribute {value_attribute}")
            data.append(element_data)
        self.properties_df = pd.DataFrame(data).set_index("GUID")
        return self.properties_df

    def get_object_placement(self, element):
        if element.ObjectPlacement:
            placement = element.ObjectPlacement
            if placement.is_a('IfcLocalPlacement'):
                relative_placement = placement.RelativePlacement
                if relative_placement:
                    location = relative_placement.Location
                    if location:
                        return {"X": location.Coordinates[0], "Y": location.Coordinates[1], "Z": location.Coordinates[2]}
        return None

    def update_properties_df(self, updated_df):
        logger.info("Starting update_properties_df method")
        logger.info(f"Number of rows in updated_df: {len(updated_df)}")
        logger.info(f"Columns in updated_df: {updated_df.columns.tolist()}")
        
        for guid in updated_df.index:
            if guid in self.properties_df.index:
                for col in updated_df.columns:
                    if col in self.properties_df.columns:
                        old_value = self.properties_df.loc[guid, col]
                        new_value = updated_df.loc[guid, col]
                        if isinstance(old_value, pd.Series) or isinstance(new_value, pd.Series):
                            if not (old_value.equals(new_value)):
                                self.properties_df.loc[guid, col] = new_value
                                logger.info(f"Updated {col} for GUID {guid}: {old_value} -> {new_value}")
                        elif pd.notna(new_value) and old_value != new_value:
                            self.properties_df.loc[guid, col] = new_value
                            logger.info(f"Updated {col} for GUID {guid}: {old_value} -> {new_value}")
                    else:
                        logger.warning(f"Column {col} not found in properties_df for GUID {guid}")
            else:
                logger.warning(f"GUID {guid} not found in properties_df")
        
        logger.info("Finished update_properties_df method")
        logger.info(f"Number of rows in properties_df after update: {len(self.properties_df)}")

    def update_properties(self, df):
        logger.info("Starting update_properties method")
        logger.info(f"Number of rows in df: {len(df)}")
        logger.info(f"Columns in df: {df.columns.tolist()}")
        logger.info(f"Sample data from df:\n{df.head().to_string()}")
        
        updated_elements = set()
        for guid, row in df.iterrows():
            logger.info(f"Processing GUID: {guid}")
            element = self.ifc_file.by_guid(guid)
            if element:
                logger.info(f"Found element with Type: {element.is_a()}")
                for prop_name, value in row.items():
                    if prop_name not in ["Type", "Name"] and pd.notna(value):
                        logger.info(f"Updating property: {prop_name} = {value}")
                        try:
                            pset_name, prop_name = prop_name.split(".", 1)
                            old_value = self._get_property_value(element, pset_name, prop_name)
                            self._update_property(element, pset_name, prop_name, value)
                            new_value = self._get_property_value(element, pset_name, prop_name)
                            logger.info(f"Updated {pset_name}.{prop_name}: {old_value} -> {new_value}")
                        except ValueError:
                            logger.warning(f"Skipping property {prop_name} due to incorrect format")
                updated_elements.add(element)
            else:
                logger.warning(f"Element with GUID {guid} not found in IFC file")
        
        logger.info(f"Total updated elements: {len(updated_elements)}")
        
        owner_history = self.ifc_file.by_type("IfcOwnerHistory")[0]
        for element in updated_elements:
            element.OwnerHistory = owner_history
        
        self.save_ifc()
        logger.info("IFC file saved with updates")

    def _update_property(self, element, pset_name, prop_name, value):
        pset = self._get_or_create_pset(element, pset_name)
        prop = self._get_or_create_property(pset, prop_name)
        
        if prop is None:
            return

        if prop.is_a("IfcPropertySingleValue"):
            prop.NominalValue = self._create_ifc_value(value)
        elif prop.is_a("IfcQuantityLength"):
            prop.LengthValue = float(value)
        elif prop.is_a("IfcQuantityArea"):
            prop.AreaValue = float(value)
        elif prop.is_a("IfcQuantityVolume"):
            prop.VolumeValue = float(value)
        elif prop.is_a("IfcQuantityCount"):
            prop.CountValue = int(float(value))

    def _update_single_property(self, element, prop_name, value):
        for definition in element.IsDefinedBy:
            if definition.is_a("IfcRelDefinesByProperties"):
                property_set = definition.RelatingPropertyDefinition
                if property_set.is_a("IfcPropertySet"):
                    for prop in property_set.HasProperties:
                        if prop.Name == prop_name:
                            if prop.is_a("IfcPropertySingleValue"):
                                prop.NominalValue = self._create_ifc_value(value)
                                return
        
        pset = self._get_or_create_pset(element, "CustomProperties")
        prop = self.ifc_file.create_entity("IfcPropertySingleValue", Name=prop_name, NominalValue=self._create_ifc_value(value))
        pset.HasProperties = list(pset.HasProperties) + [prop]

    def _get_property_value(self, element, pset_name, prop_name):
        for definition in element.IsDefinedBy:
            if definition.is_a("IfcRelDefinesByProperties"):
                if definition.RelatingPropertyDefinition.Name == pset_name:
                    for prop in definition.RelatingPropertyDefinition.HasProperties:
                        if prop.Name == prop_name:
                            if prop.is_a("IfcPropertySingleValue"):
                                return prop.NominalValue.wrappedValue if prop.NominalValue else None
        return None

    def _create_ifc_value(self, value):
        if isinstance(value, (int, float)):
            return self.ifc_file.create_entity("IfcReal", float(value))
        elif isinstance(value, bool):
            return self.ifc_file.create_entity("IfcBoolean", value)
        elif pd.isna(value):
            return None
        else:
            return self.ifc_file.create_entity("IfcLabel", str(value))

    def get_all_property_names(self):
        property_names = set()
        for element in self.ifc_file.by_type("IfcProduct"):
            for definition in element.IsDefinedBy:
                if definition.is_a("IfcRelDefinesByProperties"):
                    property_set = definition.RelatingPropertyDefinition
                    if property_set.is_a("IfcPropertySet"):
                        for property in property_set.HasProperties:
                            if property.is_a("IfcPropertySingleValue"):
                                property_names.add(f"{property_set.Name}.{property.Name}")
                    elif property_set.is_a("IfcElementQuantity"):
                        for quantity in property_set.Quantities:
                            if quantity.is_a("IfcQuantityLength") or quantity.is_a("IfcQuantityArea") or quantity.is_a("IfcQuantityVolume") or quantity.is_a("IfcQuantityCount"):
                                property_names.add(f"{property_set.Name}.{quantity.Name}")
        return sorted(list(property_names))

    def _get_or_create_pset(self, element, pset_name):
        for definition in element.IsDefinedBy:
            if definition.is_a("IfcRelDefinesByProperties"):
                if definition.RelatingPropertyDefinition.Name == pset_name:
                    return definition.RelatingPropertyDefinition
        
        pset = self.ifc_file.create_entity("IfcPropertySet")
        pset.GlobalId = str(uuid.uuid4())
        pset.OwnerHistory = element.OwnerHistory
        pset.Name = pset_name
        pset.Description = None
        pset.HasProperties = []

        rel = self.ifc_file.create_entity("IfcRelDefinesByProperties")
        rel.GlobalId = str(uuid.uuid4())
        rel.OwnerHistory = element.OwnerHistory
        rel.RelatedObjects = [element]
        rel.RelatingPropertyDefinition = pset

        return pset

    def _get_or_create_property(self, pset, prop_name):
        if pset.is_a("IfcPropertySet"):
            for prop in pset.HasProperties:
                if prop.Name == prop_name:
                    return prop
            
            prop = self.ifc_file.create_entity("IfcPropertySingleValue", Name=prop_name, NominalValue=None)
            pset.HasProperties = list(pset.HasProperties) + [prop]
            return prop
        elif pset.is_a("IfcElementQuantity"):
            for quantity in pset.Quantities:
                if quantity.Name == prop_name:
                    return quantity
            
            quantity = self.ifc_file.create_entity("IfcQuantityLength", Name=prop_name, LengthValue=0.0)
            pset.Quantities = list(pset.Quantities) + [quantity]
            return quantity
        else:
            print(f"Unexpected property set type: {pset.is_a()}")
            return None

    def save_ifc(self):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ifc') as tmp_file:
                self.ifc_file.write(tmp_file.name)
                tmp_file.seek(0)
                ifc_data = tmp_file.read()
            os.unlink(tmp_file.name)
            return ifc_data
        except Exception as e:
            logger.error(f"Error saving IFC file: {str(e)}")
            return None

    def add_new_property(self, pset_name, prop_name, prop_type="IfcText", default_value=None):
        for element in self.ifc_file.by_type("IfcProduct"):
            pset = self._get_or_create_pset(element, pset_name)
            prop = self.ifc_file.create_entity("IfcPropertySingleValue", Name=prop_name)
            if default_value is not None:
                prop.NominalValue = self.ifc_file.create_entity(prop_type, default_value)
            pset.HasProperties = list(pset.HasProperties) + [prop]
        return prop

    def get_debug_logs(self):
        return "\n".join(self.debug_logs)